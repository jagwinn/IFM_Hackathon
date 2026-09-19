"""Bounded, request-local lifecycle with a dependency-aware scheduler."""

import asyncio
from dataclasses import asdict
import logging
import time
from uuid import uuid4

from .config import Limits
from .schemas import BudgetError, Plan, RelayError, Reply, RunResult, Task, ValidationError
from .validators import validate_result

log = logging.getLogger(__name__)


class RelayEngine:
    def __init__(self, provider, limits=None):
        self.provider = provider
        self.limits = limits or Limits()

    async def run(self, goal, sources, *, accept_local=False, emit=None):
        if not isinstance(goal, str) or not 1 <= len(goal) <= 8000:
            raise RelayError("Goal must be text of at most 8000 characters")
        if (not isinstance(sources, dict) or not sources or len(sources) > self.limits.max_tasks
                or any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in sources.items())
                or sum(len(v) for v in sources.values()) > 32000):
            raise RelayError("Sources must be nonempty text within the prototype's 32000-character limit")
        run = _Run(self.provider, self.limits, emit)
        try:
            async with asyncio.timeout(self.limits.run_timeout):
                await run.event("run_started", "Starting relay")
                if getattr(self.provider, "scenario", "") in ("repair", "escalate"):
                    await run.event("fault_injection", "Demo intentionally injects an invalid worker quote")
                result = await run.execute(goal, dict(sources), accept_local)
                await run.event("run_completed", "Relay complete", done=True, metrics=run.metrics())
                return RunResult(run.id, result, run.results, run.metrics())
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
    def __init__(self, provider, limits, emit):
        self.provider, self.limits, self.emit = provider, limits, emit
        self.id, self.seq = uuid4().hex, 0
        self.started = time.monotonic()
        self.calls, self.results = [], {}
        self.cloud_count = self.cloud_workers = 0
        self.local_slots = asyncio.Semaphore(limits.local_concurrency)
        self.event_lock = asyncio.Lock()

    def metrics(self):
        return {
            "simulated": self.provider.simulated,
            "elapsed_ms": round((time.monotonic() - self.started) * 1000),
            "local_calls": sum(c["tier"] == "local" for c in self.calls),
            "cloud_calls": self.cloud_count,
            "completed_tasks": list(self.results),
            "calls": [dict(c) for c in self.calls],
        }

    async def event(self, name, description, *, done=False, **fields):
        self.seq += 1
        prefix = "[SIMULATED] " if self.provider.simulated else ""
        event = {"type": "status", "data": {
            "action": "horizon_relay", "description": prefix + description, "done": done,
            "relay": {"version": 1, "run_id": self.id, "seq": self.seq,
                      "event": name, "simulated": self.provider.simulated, **fields},
        }}
        if self.emit:
            # Open WebUI persists history using a read/append/write operation.
            # Concurrent callbacks can otherwise overwrite a sibling's status.
            async with self.event_lock:
                try:
                    async with asyncio.timeout(1):
                        await self.emit(event)
                except Exception:
                    # UI delivery failure must not duplicate provider calls or lose the answer.
                    log.warning("Relay progress delivery failed")

    async def call(self, tier, operation, payload):
        if tier == "cloud":
            if not self.limits.cloud_enabled:
                raise BudgetError("Cloud is disabled; this request requires the planner")
            reserve = 0 if operation == "synthesize" else 1
            if self.cloud_count >= self.limits.cloud_calls - reserve:
                raise BudgetError("Cloud call budget exhausted (final synthesis reserved)")
            if operation == "work":
                if self.cloud_workers >= self.limits.cloud_worker_calls:
                    raise BudgetError("Cloud worker budget exhausted")
                self.cloud_workers += 1
            self.cloud_count += 1
        record = {"tier": tier, "operation": operation, "task_id": payload.get("task", {}).get("id"),
                  "model": None, "prompt_tokens": None, "completion_tokens": None,
                  "outcome": "started", "elapsed_ms": 0}
        self.calls.append(record)
        start = time.monotonic()
        try:
            async with asyncio.timeout(self.limits.call_timeout):
                reply = await self.provider.complete(tier, operation, payload)
            if not isinstance(reply, Reply) or not reply.model:
                raise RelayError("Invalid provider response envelope")
            record.update(model=reply.model, prompt_tokens=reply.prompt_tokens,
                          completion_tokens=reply.completion_tokens, outcome="completed")
            return reply.value
        except asyncio.CancelledError:
            record["outcome"] = "cancelled"
            raise
        except Exception:
            record["outcome"] = "failed"
            raise
        finally:
            record["elapsed_ms"] = round((time.monotonic() - start) * 1000)

    async def execute(self, goal, sources, accept_local):
        await self.event("local_attempt", "Trying the local model", tier="local")
        candidate = await self.call("local", "attempt", {"goal": goal, "sources": sources})
        # Caller opts into a narrowly checkable extraction; confidence never grants acceptance.
        if accept_local and len(sources) == 1:
            local_task = Task("local_answer", goal, "local", tuple(f"sources.{k}" for k in sources),
                              (), "report_extraction", ("required_fields", "source_quotes_match"))
            try:
                validate_result(local_task, candidate, {f"sources.{k}": v for k, v in sources.items()})
                if candidate["quote"] != next(iter(sources.values())):
                    raise ValidationError("Verbatim local answer must contain the full report")
            except ValidationError:
                pass
            else:
                self.results["local_answer"] = candidate
                await self.event("local_accepted", "Source evidence checked; answered locally", tier="local")
                return self.label(f'**{candidate["report_id"]}**: {candidate["quote"]}')
        await self.event("escalated", "Request needs cloud planning", tier="cloud")
        error = None
        plan = None
        for attempt in (1, 2):
            raw = await self.call("cloud", "plan", {"goal": goal, "sources": sources, "validation_error": error})
            try:
                plan = Plan.parse(raw, sources, self.limits.max_tasks)
                break
            except ValidationError as exc:
                error = str(exc)
                await self.event("plan_invalid", "Planner returned an invalid graph", attempt=attempt, error=error)
        if plan is None:
            raise ValidationError("Planner returned an invalid graph twice")
        await self.event("plan_created", f"Validated plan with {len(plan.tasks)} subtasks",
                         tasks=[asdict(t) for t in plan.tasks])
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
        await self.event("synthesizing", "Cloud model assembling accepted results", tier="cloud")
        answer = await self.call("cloud", "synthesize", {
            "goal": goal, "instruction": plan.final_instruction, "results": self.results,
        })
        if not isinstance(answer, str) or not answer.strip():
            raise ValidationError("Synthesis returned no answer")
        return self.label(answer)

    def label(self, answer):
        if self.provider.simulated:
            return "**SIMULATED DEMO — built-in bug reports; no real model or API calls.**\n\n" + answer
        return answer

    async def worker(self, task, sources, goal):
        inputs = {}
        for ref in task.input_refs:
            prefix, key = ref.split(".", 1)
            inputs[ref] = sources[key] if prefix == "sources" else self.results[key]
        errors = []
        tiers = ("local", "local", "cloud") if task.preferred_tier == "local" else ("cloud",)
        for attempt, tier in enumerate(tiers, 1):
            async def perform():
                await self.event("task_started", f"{task.id}: {tier} attempt {attempt}",
                                 task_id=task.id, tier=tier, attempt=attempt)
                return await self.call(tier, "work", {"task": asdict(task), "inputs": inputs,
                                                     "goal": goal, "attempt": attempt, "errors": list(errors)})
            if tier == "cloud" and attempt > 1:
                await self.event("task_escalated", f"Escalating only {task.id}", task_id=task.id, tier=tier)
            if tier == "local":
                async with self.local_slots:
                    value = await perform()
            else:
                value = await perform()
            try:
                validate_result(task, value, inputs)
            except ValidationError as exc:
                errors.append(str(exc))
                await self.event("task_invalid", f"{task.id}: result failed checks", task_id=task.id,
                                 tier=tier, attempt=attempt, error=str(exc))
                continue
            # Retain accepted work even if a sibling fails later in this batch.
            self.results[task.id] = value
            await self.event("task_completed", f"{task.id}: checks passed", task_id=task.id,
                             tier=tier, attempt=attempt)
            return value
        raise ValidationError(f"Subtask {task.id} failed validation; dependent tasks were not run")
