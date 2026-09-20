"""Orchestration mechanics: local models answer and are judged, smallest first; if none is trusted,
the cloud answers directly or plans, delegates checked subtasks and synthesizes.

What to decide (trust a local answer? retry? escalate?) comes from policy.RoutingPolicy. This module
carries those decisions out and enforces limits, deadlines, cancellation, metrics and progress events.
"""

import asyncio
from dataclasses import asdict, dataclass, replace
import json
import logging
import time
from uuid import uuid4

from .budget import CloudBudget
from .errors import BudgetError, RelayError, ValidationError
from .events import EventCallback, RelayEvent
from .plan import Plan
from .policy import (Critique, RoutingPolicy, answer_exact_extraction, read_assessment, read_critique,
                     request_parts, task_complexity)
from .providers import Provider, Reply, available_tiers
from .results import split_judgment, validate_result
from .settings import Limits
from .tokens import answer_token_stats

log = logging.getLogger(__name__)

LIVE_INTERVAL = 1.5  # seconds between live-output events while a model writes
LIVE_LIMIT = 6000  # characters of thinking/answer kept for display

SIMULATED_LABEL = "**SIMULATED DEMO — built-in bug reports; no real model or API calls.**\n\n"


@dataclass(frozen=True)
class RunResult:
    run_id: str
    content: str  # the final answer (Markdown)
    results: dict  # accepted subtask results by task ID
    metrics: dict  # simulated, route, verdicts, elapsed_ms, local_calls, cloud_calls, completed_tasks, calls

    def summary(self) -> str:
        """One-line call and token count, e.g. for a chat footer."""
        calls = self.metrics["calls"]
        if self.metrics["simulated"]:
            usage = "unavailable (simulation)"
        elif all(c["prompt_tokens"] is not None and c["completion_tokens"] is not None for c in calls):
            usage = str(sum(c["prompt_tokens"] + c["completion_tokens"] for c in calls))
        else:
            usage = "unavailable for some calls"
        return (f'Relay: {self.metrics["local_calls"]} local calls · '
                f'{self.metrics["cloud_calls"]} cloud calls · tokens: {usage}.')


class RelayEngine:
    def __init__(self, provider: Provider, limits: Limits | None = None, policy: RoutingPolicy | None = None):
        self.provider = provider
        self.limits = limits or Limits()
        self.policy = policy or RoutingPolicy()

    async def run(self, goal: str, sources: dict[str, str], *, local_extract: bool = False,
                  emit: EventCallback | None = None, on_delta=None) -> RunResult:
        """`emit` receives progress events; `on_delta(call, text)` receives each chunk a model writes, where
        call describes the step (tier, operation, task_id, attempt)."""
        if not isinstance(goal, str) or not 1 <= len(goal) <= 8000:
            raise RelayError("Goal must be text of at most 8000 characters")
        if (not isinstance(sources, dict) or not sources or len(sources) > self.limits.max_tasks
                or any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in sources.items())
                or sum(len(v) for v in sources.values()) > 32000):
            raise RelayError("Sources must be nonempty text within the prototype's 32000-character limit")
        run = _Run(self.provider, self.limits, self.policy, emit, on_delta)
        try:
            async with asyncio.timeout(self.limits.run_timeout):
                locations = await _locations(self.provider)
                await run.event("run_started", "Starting relay", goal=goal[:2000], tiers=[
                    {"id": t, "model": run.models.get(t, t), "location": locations.get(t, "")} for t in run.tiers])
                if notice := getattr(self.provider, "notice", None):
                    await run.event("provider_notice", notice)
                content = await run.execute(goal, dict(sources), local_extract)
                await run.event("run_completed", "Relay complete", done=True, metrics=run.metrics())
                return RunResult(run.id, content, run.results, run.metrics())
        except asyncio.CancelledError:
            await asyncio.shield(run.event("run_cancelled", "Relay cancelled", done=True, metrics=run.metrics()))
            raise
        except TimeoutError as exc:
            await run.event("run_failed", "Relay deadline exceeded", done=True, metrics=run.metrics())
            raise RelayError("Relay deadline exceeded; no complete answer was produced") from exc
        except Exception as exc:
            await run.event("run_failed", "Relay stopped before completion", done=True, metrics=run.metrics())
            if isinstance(exc, RelayError):
                raise
            raise RelayError("Relay provider failed; no complete answer was produced") from exc


class _Run:
    """State for one request. Nothing is shared between runs."""

    def __init__(self, provider, limits, policy, emit, on_delta=None):
        self.provider, self.limits, self.policy, self.emit = provider, limits, policy, emit
        self.on_delta = on_delta
        self.id, self.seq = uuid4().hex, 0
        self.started = time.monotonic()
        self.calls, self.results = [], {}
        self.tiers = available_tiers(provider)  # smallest first
        self.models = dict(getattr(provider, "models", None) or {})
        self.budget = CloudBudget(limits)
        self.route = None  # the tier whose answer was returned: a local tier or "cloud"
        self.verdicts = []  # policy.Verdict for each local model that answered
        self.skipped = []  # local tiers the policy skipped for this request
        self.local_slots = asyncio.Semaphore(limits.local_concurrency)
        self.event_lock = asyncio.Lock()

    async def execute(self, goal, sources, local_extract):
        text = _task_text(goal, sources)
        if local_extract:
            answer = await self.try_extract(goal, sources)
            if answer is not None:
                self.route = "local"
                return self.label(answer)
        complexity = task_complexity(text)
        for tier in (t for t in self.tiers if t != "cloud"):
            if self.policy.skips(tier, self.tiers, complexity):
                self.skipped.append(tier)
                await self.event("local_skipped", f"{self.models.get(tier, tier)} skipped: request looks hard", tier=tier,
                                 complexity=round(complexity, 4), threshold=self.policy.skip_small_above)
                continue
            verdict = await self.judge_locally(tier, text)
            self.verdicts.append(verdict)
            if not verdict.escalate:
                self.route = tier
                await self.event("local_accepted", f"{self.models.get(tier, tier)} answer trusted", tier=tier,
                                 reason=verdict.reason)
                return self.label(verdict.answer)
        self.route = "cloud"
        context = [{"tier": v.tier, "model": self.models.get(v.tier, v.tier), "answer": v.answer[:4000],
                    "reason": f"{v.reason}. Critic: {v.critique.critique}"} for v in self.verdicts]
        plans = self.policy.plans(text)
        await self.event("escalated", "Local answers not trusted; the cloud takes over", tier="cloud",
                         mode="plan" if plans else "direct", parts=request_parts(text), length=len(text),
                         reason=self.verdicts[-1].reason if self.verdicts else None)
        if not plans:
            await self.event("answering", "Cloud model answering", tier="cloud")
            answer = await self.call("cloud", "answer", {"task": text, "local_attempts": context})
            if not isinstance(answer, str) or not answer.strip():
                raise ValidationError("Cloud answer was empty")
            return self.label(answer)
        plan = await self.make_plan(goal, sources, context)
        await self.run_tasks(plan, goal, sources)
        await self.event("synthesizing", "Cloud model assembling accepted results", tier="cloud")
        answer = await self.call("cloud", "synthesize", {
            "goal": goal, "sources": sources, "instruction": plan.final_instruction, "results": self.results,
        })
        if not isinstance(answer, str) or not answer.strip():
            raise ValidationError("Synthesis returned no answer")
        return self.label(answer)

    async def try_extract(self, goal, sources):
        """/local-extract: an exact copy of one source, accepted only when it checks out exactly."""
        await self.event("local_attempt", "Trying an exact local extraction", tier="local", model=self.models.get("local"))
        candidate = await self.call("local", "attempt", {"goal": goal, "sources": sources, "local_extract": True})
        answer = answer_exact_extraction(goal, sources, candidate)
        if answer is None:
            await self.event("extract_failed", "Extraction did not match the source exactly", tier="local")
            return None
        self.results["local_answer"] = candidate
        await self.event("local_accepted", "Source evidence checked; answered locally", tier="local")
        return answer

    async def judge_locally(self, tier, text):
        """One local model answers (policy.local_samples times) and a critic checks it; the policy decides."""
        model = self.models.get(tier, tier)
        await self.event("local_solve", f"{model} is answering", tier=tier, model=model)
        samples = []
        for i in range(self.policy.local_samples):
            token_stats = None
            try:
                value, record = await self.call_with_record(tier, "solve", {
                    "task": text, "temperature": round(0.25 + 0.15 * i, 2), "attempt": i + 1, "for_tier": tier})
                token_stats = record["token_stats"]
            except RelayError as exc:
                # A local model that cannot answer (timeout, token limit, bad reply) is simply not trusted.
                value = {"answer": "", "confidence": 0.0, "task_type": "failed", "difficulty": "high",
                         "needs_escalation": True, "reason": f"The call failed: {exc}"}
            sample = read_assessment(value, token_stats)
            samples.append(sample)
            await self.event("local_sample", f"{model} answer {i + 1}: confidence {sample.confidence:.2f}", tier=tier,
                             attempt=i + 1, assessment=asdict(sample))
            if sample.needs_escalation:
                break  # no point sampling again once the model asks for help
        critic = self.policy.critic_tier(tier, self.tiers)
        if not self.policy.use_critic:
            critique = Critique(True, 0.0, "Critic disabled by policy.", skipped=True)
        elif any(s.needs_escalation for s in samples):
            critique = Critique(False, 1.0, "Critic skipped: the model already asked for a larger model.", skipped=True)
        else:
            await self.event("local_critique", f"{self.models.get(critic, critic)} is checking the answer", tier=tier,
                             critic_tier=critic, model=self.models.get(critic, critic))
            try:
                raw = await self.call(critic, "critique", {"task": text, "answer": samples[0].answer,
                                                           "attempt": len(samples) + 1, "for_tier": tier})
                critique = replace(read_critique(raw), tier=critic)
            except RelayError as exc:
                critique = Critique(False, 1.0, f"The critic call failed: {exc}", tier=critic)
        verdict = self.policy.decide(tier, text, samples, critique)
        await self.event("local_verdict", verdict.reason, tier=tier, verdict=verdict.to_dict(), weights=dict(self.policy.weights))
        return verdict

    async def make_plan(self, goal, sources, local_attempts=()):
        error = None
        for attempt in range(1, self.policy.plan_attempts + 1):
            raw = await self.call("cloud", "plan", {"goal": goal, "sources": sources, "validation_error": error,
                                                    "tiers": list(self.tiers), "local_attempts": list(local_attempts),
                                                    "attempt": attempt})
            try:
                plan = Plan.parse(raw, sources, self.limits.max_tasks, self.tiers)
            except ValidationError as exc:
                error = str(exc)
                await self.event("plan_invalid", "Planner returned an invalid graph", attempt=attempt, error=error)
                continue
            await self.event("plan_created", f"Validated plan with {len(plan.tasks)} subtasks",
                             tasks=[asdict(t) for t in plan.tasks], final_instruction=plan.final_instruction)
            return plan
        raise ValidationError(f"Planner returned an invalid graph {self.policy.plan_attempts} times")

    async def run_tasks(self, plan, goal, sources):
        """Run every task whose dependencies are done, in parallel batches, until the graph is complete."""
        pending = {t.id: t for t in plan.tasks}
        while pending:
            ready = [t for t in pending.values() if set(t.depends_on) <= self.results.keys()]
            if not ready:
                raise ValidationError("No executable task remains")
            jobs = [asyncio.create_task(self.worker(t, sources, goal)) for t in ready]
            try:
                values = await asyncio.gather(*jobs)
                for task, value in zip(ready, values):
                    self.results[task.id] = value
                    del pending[task.id]
            finally:
                for job in jobs:
                    if not job.done():
                        job.cancel()
                await asyncio.gather(*jobs, return_exceptions=True)

    async def worker(self, task, sources, goal):
        inputs = {}
        for ref in task.input_refs:
            prefix, key = ref.split(".", 1)
            inputs[ref] = sources[key] if prefix == "sources" else self.results[key]
        errors = []
        previous = task.preferred_tier
        for attempt, tier in enumerate(self.policy.worker_tiers(task, self.tiers), 1):
            async def perform():
                await self.event("task_started", f"{task.id}: {tier} attempt {attempt}",
                                 task_id=task.id, tier=tier, attempt=attempt, model=self.models.get(tier))
                return await self.call_with_record(tier, "work", {"task": asdict(task), "inputs": inputs, "goal": goal,
                                                                  "attempt": attempt, "errors": list(errors)})
            if tier != previous:
                await self.event("task_escalated", f"Escalating only {task.id} to {tier}", task_id=task.id,
                                 tier=tier, from_tier=previous, attempt=attempt, reason=errors[-1] if errors else None)
            previous = tier
            try:
                if tier != "cloud":
                    async with self.local_slots:
                        value, record = await perform()
                else:
                    value, record = await perform()
            except BudgetError:
                raise
            except RelayError as exc:
                # A call that fails (token limit, timeout, unreadable reply) counts as a failed attempt:
                # the subtask retries or escalates like any result that fails its checks.
                errors.append(str(exc))
                await self.event("task_invalid", f"{task.id}: call failed", task_id=task.id, tier=tier,
                                 attempt=attempt, error=str(exc), judgment=None)
                continue
            try:
                validate_result(task.result_type, value, inputs)
            except ValidationError as exc:
                errors.append(str(exc))
                await self.event("task_invalid", f"{task.id}: result failed checks", task_id=task.id,
                                 tier=tier, attempt=attempt, error=str(exc), judgment=record["judgment"])
                continue
            # Retain accepted work even if a sibling fails later in this batch.
            self.results[task.id] = value
            check_label = "output format checked" if task.result_type == "task_answer" else "checks passed"
            await self.event("task_completed", f"{task.id}: {check_label}", task_id=task.id, tier=tier,
                             attempt=attempt, judgment=record["judgment"], result=_preview(value))
            return value
        raise ValidationError(f"Subtask {task.id} failed validation; dependent tasks were not run")

    async def call(self, tier, operation, payload):
        value, _ = await self.call_with_record(tier, operation, payload)
        return value

    async def call_with_record(self, tier, operation, payload):
        """One model call. Returns (value, call record); the model's judgment is split off into the record."""
        if tier == "cloud":
            self.budget.spend(operation)
        task = payload.get("task")
        record = {"tier": tier, "operation": operation, "task_id": task.get("id") if isinstance(task, dict) else None,
                  "attempt": payload.get("attempt"), "for_tier": payload.get("for_tier"), "model": None, "prompt_tokens": None,
                  "completion_tokens": None, "outcome": "started", "elapsed_ms": 0, "judgment": None,
                  "token_stats": None, "output": None}
        self.calls.append(record)
        start = time.monotonic()
        live = {"text": "", "shown": 0}
        watcher = None
        try:
            async with asyncio.timeout(self.limits.call_timeout):
                if (self.emit or self.on_delta) and getattr(self.provider, "streams", False):
                    def deliver(text, kind="output"):
                        live["text"] += text
                        if self.on_delta:
                            self.on_delta({"tier": tier, "operation": operation, "task_id": record["task_id"],
                                           "attempt": record["attempt"], "for_tier": record["for_tier"],
                                           "model": self.models.get(tier, tier), "kind": kind}, text)
                    if self.emit:
                        watcher = asyncio.create_task(self.follow(record, live))
                    reply = await self.provider.complete(tier, operation, payload, on_delta=deliver)
                else:
                    reply = await self.provider.complete(tier, operation, payload)
            if not isinstance(reply, Reply) or not reply.model:
                raise RelayError("Invalid provider response envelope")
            value, judgment = split_judgment(reply.value) if operation in ("attempt", "work") else (reply.value, None)
            record.update(model=reply.model, prompt_tokens=reply.prompt_tokens,
                          completion_tokens=reply.completion_tokens, outcome="completed", judgment=judgment,
                          token_stats=answer_token_stats(reply.tokens),
                          output={"thinking": reply.thinking[:LIVE_LIMIT], "answer": reply.text[:LIVE_LIMIT]})
        except asyncio.CancelledError:
            record.update(outcome="cancelled", elapsed_ms=round((time.monotonic() - start) * 1000))
            raise
        except Exception:
            record.update(outcome="failed", elapsed_ms=round((time.monotonic() - start) * 1000),
                          output={"thinking": "", "answer": live["text"][:LIVE_LIMIT]})
            await self.event("call_finished", f"{tier} {operation} call failed", **record)
            raise
        finally:
            if watcher:
                watcher.cancel()
        record["elapsed_ms"] = round((time.monotonic() - start) * 1000)
        await self.event("call_finished", f"{tier} {operation} call finished", **record)
        return value, record

    async def follow(self, record, live):
        """While a call streams, publish what the model has written so far, a few times a second at most."""
        try:
            while True:
                await asyncio.sleep(LIVE_INTERVAL)
                if len(live["text"]) == live["shown"]:
                    continue
                live["shown"] = len(live["text"])
                await self.event("call_output", "Model is writing", tier=record["tier"], operation=record["operation"],
                                 task_id=record["task_id"], for_tier=record["for_tier"], attempt=record["attempt"],
                                 output=live["text"][-LIVE_LIMIT:], written=len(live["text"]))
        except asyncio.CancelledError:
            pass

    async def event(self, name, message, *, done=False, **data):
        self.seq += 1
        if not self.emit:
            return
        event = RelayEvent(self.id, self.seq, name, message, done, self.provider.simulated, data)
        # One callback at a time, in order: front ends that persist history by read/append/write
        # would otherwise overwrite a sibling's event.
        async with self.event_lock:
            try:
                async with asyncio.timeout(1):
                    await self.emit(event)
            except Exception:
                # Display failure must not duplicate provider calls or lose the answer.
                log.warning("Relay progress delivery failed")

    def metrics(self):
        return {
            "simulated": self.provider.simulated,
            "route": self.route,
            "verdicts": [v.to_dict() for v in self.verdicts],
            "skipped": list(self.skipped),
            "elapsed_ms": round((time.monotonic() - self.started) * 1000),
            "local_calls": sum(c["tier"] != "cloud" for c in self.calls),
            "cloud_calls": self.budget.calls,
            "completed_tasks": list(self.results),
            "calls": [dict(c) for c in self.calls],
        }

    def label(self, answer):
        return SIMULATED_LABEL + answer if self.provider.simulated else answer


def _task_text(goal, sources):
    """What a local model is asked to answer: the request itself, with any earlier conversation as context."""
    if list(sources) == ["conversation"]:
        conversation = sources["conversation"]
        if conversation.strip() == f"user: {goal}":
            return goal
        return f"{conversation}\n\nRespond to the last user message."
    return goal + "\n\n" + "\n\n".join(f"[{key}]\n{text}" for key, text in sources.items())


async def _locations(provider) -> dict:
    """Optional provider hook describing where each tier's model runs. Display only; failures are ignored."""
    describe = getattr(provider, "locations", None)
    if describe is None:
        return {}
    try:
        async with asyncio.timeout(3):
            found = await describe()
        return found if isinstance(found, dict) else {}
    except Exception:
        log.info("Model locations unavailable")
        return {}


def _preview(value, limit=600):
    """Short text form of an accepted result for progress displays."""
    if isinstance(value, dict) and set(value) == {"answer"}:
        text = value["answer"]
    else:
        text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit - 1] + "…"
