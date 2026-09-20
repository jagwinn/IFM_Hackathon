import asyncio
from dataclasses import replace
import unittest

from horizon_relay import (BudgetError, Limits, RelayEngine, RelayError, Reply, RoutingPolicy, SimulatedProvider,
                           ValidationError)
from horizon_relay.plan import Plan
from horizon_relay.providers.simulated import demo_sources


PLANS = RoutingPolicy(cloud_mode="plan")  # these tests exercise planning and delegation


class RelayTests(unittest.IsolatedAsyncioTestCase):
    async def run_demo(self, scenario="standard", limits=None, provider=None, policy=None):
        events = []
        async def emit(event):
            events.append(event)
        sources = demo_sources()
        if scenario == "local":
            sources = {"report_1": sources["report_1"]}
        result = await RelayEngine(provider or SimulatedProvider(scenario), limits,
                                   policy or RoutingPolicy(cloud_mode="plan")).run(
            "Extract report verbatim" if scenario == "local" else "Prioritize bug reports",
            sources, local_extract=scenario == "local", emit=emit,
        )
        return result, events

    async def test_standard_delegates_and_synthesizes(self):
        result, events = await self.run_demo()
        self.assertEqual(result.metrics["cloud_calls"], 2)
        self.assertEqual(result.metrics["local_calls"], 4)
        self.assertEqual(len(result.results), 3)
        self.assertIn("SIMULATED", result.content)
        completed = [e.data.get("task_id") for e in events if e.name == "task_completed"]
        self.assertEqual(completed[-1], "group_reports")
        self.assertEqual([e.seq for e in events], list(range(1, len(events) + 1)))
        self.assertTrue(events[-1].done)
        self.assertTrue(all(c["prompt_tokens"] is None for c in result.metrics["calls"]))

    async def test_local_needs_no_cloud(self):
        result, _ = await self.run_demo("local", Limits(cloud_enabled=False, cloud_calls=0))
        self.assertEqual(result.metrics["cloud_calls"], 0)
        self.assertEqual(result.metrics["local_calls"], 1)

    async def test_invalid_local_candidate_escalates(self):
        class InvalidCandidate(SimulatedProvider):
            async def complete(self, tier, operation, payload):
                if operation == "attempt":
                    return Reply({"report_id": "report_1", "quote": "Invented"}, "fake")
                return await super().complete(tier, operation, payload)
        result, _ = await self.run_demo("local", provider=InvalidCandidate())
        self.assertEqual(result.metrics["cloud_calls"], 2)

    async def test_one_local_repair(self):
        result, events = await self.run_demo("repair")
        self.assertEqual(result.metrics["cloud_calls"], 2)
        self.assertEqual(result.metrics["local_calls"], 5)
        names = [e.name for e in events]
        self.assertIn("task_invalid", names)
        self.assertNotIn("task_escalated", names)
        self.assertIn("provider_notice", names)

    async def test_partial_quote_cannot_satisfy_verbatim_local_request(self):
        class PartialQuote(SimulatedProvider):
            async def complete(self, tier, operation, payload):
                if operation == "attempt":
                    return Reply({"report_id": "report_1", "quote": "Checkout fails"}, "fake")
                return await super().complete(tier, operation, payload)
        result, _ = await self.run_demo("local", provider=PartialQuote())
        self.assertEqual(result.metrics["cloud_calls"], 2)

    async def test_completed_sibling_is_retained_on_failure(self):
        accepted = asyncio.Event()
        events = []
        class FailedSibling(SimulatedProvider):
            async def complete(self, tier, operation, payload):
                if operation == "work" and payload["task"]["id"] == "extract_report_1":
                    await accepted.wait()
                    raise RuntimeError("Provider unavailable")
                return await super().complete(tier, operation, payload)
        async def emit(event):
            events.append(event)
            if event.name == "task_completed" and event.data["task_id"] == "extract_report_2":
                accepted.set()
        with self.assertRaises(RelayError):
            await RelayEngine(FailedSibling(), Limits(local_concurrency=2), PLANS).run("Goal", demo_sources(), emit=emit)
        self.assertIn("extract_report_2", events[-1].data["metrics"]["completed_tasks"])

    async def test_only_failed_task_escalates(self):
        result, _ = await self.run_demo("escalate")
        cloud_workers = [c for c in result.metrics["calls"] if c["tier"] == "cloud" and c["operation"] == "work"]
        self.assertEqual([c["task_id"] for c in cloud_workers], ["extract_report_1"])
        self.assertEqual(result.metrics["cloud_calls"], 3)

    async def test_budget_reserves_synthesis(self):
        with self.assertRaises(BudgetError):
            await self.run_demo("escalate", Limits(cloud_calls=2))

    async def test_failed_worker_call_escalates_instead_of_stopping(self):
        class Truncating(SimulatedProvider):
            async def complete(self, tier, operation, payload):
                if operation == "work" and tier == "local" and payload["task"]["id"] == "extract_report_1":
                    raise RelayError("Local response was empty or reached its token limit")
                return await super().complete(tier, operation, payload)
        result, events = await self.run_demo(provider=Truncating())
        cloud_work = [c["task_id"] for c in result.metrics["calls"] if c["tier"] == "cloud" and c["operation"] == "work"]
        self.assertEqual(cloud_work, ["extract_report_1"])
        self.assertIn("token limit", [e.data.get("error", "") for e in events if e.name == "task_invalid"][0])

    async def test_cloud_worker_budget(self):
        with self.assertRaises(BudgetError):
            await self.run_demo("escalate", Limits(cloud_worker_calls=0))

    async def test_cloud_disabled(self):
        with self.assertRaises(BudgetError):
            await self.run_demo(limits=Limits(cloud_enabled=False))

    async def test_failed_task_blocks_dependents(self):
        operations = []
        class AlwaysInvalid(SimulatedProvider):
            async def complete(self, tier, operation, payload):
                task = payload.get("task")
                operations.append((operation, task.get("id") if isinstance(task, dict) else None))
                if operation == "work" and payload["task"]["id"] == "extract_report_1":
                    return Reply({}, "fake")
                return await super().complete(tier, operation, payload)
        with self.assertRaises(ValidationError):
            await self.run_demo(provider=AlwaysInvalid())
        self.assertNotIn(("work", "group_reports"), operations)
        self.assertFalse(any(op == "synthesize" for op, _ in operations))

    async def test_plan_repairs_once_then_stops(self):
        calls = []
        class BadPlanner(SimulatedProvider):
            async def complete(self, tier, operation, payload):
                calls.append(operation)
                if operation == "plan":
                    return Reply({"tasks": []}, "fake")
                return await super().complete(tier, operation, payload)
        with self.assertRaises(ValidationError):
            await self.run_demo(provider=BadPlanner())
        self.assertEqual(calls, ["solve", "plan", "plan"])

    async def test_plan_repair_recovers(self):
        class RepairPlanner(SimulatedProvider):
            async def complete(self, tier, operation, payload):
                if operation == "plan" and payload["validation_error"] is None:
                    return Reply({}, "fake")
                return await super().complete(tier, operation, payload)
        result, _ = await self.run_demo(provider=RepairPlanner())
        self.assertEqual(result.metrics["cloud_calls"], 3)

    async def test_parallel_workers_respect_limit(self):
        active = maximum = 0
        class Observed(SimulatedProvider):
            async def complete(self, tier, operation, payload):
                nonlocal active, maximum
                if operation == "work" and tier == "local":
                    active += 1
                    maximum = max(maximum, active)
                    try:
                        await asyncio.sleep(0.01)
                        return await super().complete(tier, operation, payload)
                    finally:
                        active -= 1
                return await super().complete(tier, operation, payload)
        await self.run_demo(provider=Observed(), limits=Limits(local_concurrency=2))
        self.assertEqual(maximum, 2)
        maximum = 0
        await self.run_demo(provider=Observed(), limits=Limits(local_concurrency=1))
        self.assertEqual(maximum, 1)

    async def test_cancellation_closes_workers_and_starts_no_more_calls(self):
        worker_started = asyncio.Event()
        active = 0
        calls = []
        events = []
        class BlockedWorker(SimulatedProvider):
            async def complete(self, tier, operation, payload):
                nonlocal active
                calls.append(operation)
                if operation == "work":
                    active += 1
                    worker_started.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        active -= 1
                return await super().complete(tier, operation, payload)
        async def emit(event):
            events.append(event)
        job = asyncio.create_task(RelayEngine(BlockedWorker(), Limits(local_concurrency=2), PLANS).run("Goal", demo_sources(), emit=emit))
        await asyncio.wait_for(worker_started.wait(), 1)
        job.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await job
        self.assertEqual(active, 0)
        count = len(calls)
        await asyncio.sleep(0)
        self.assertEqual(len(calls), count)
        self.assertNotIn("synthesize", calls)
        self.assertEqual(events[-1].name, "run_cancelled")

    async def test_timeout_closes_provider(self):
        with self.assertRaises(RelayError):
            await self.run_demo(provider=SimulatedProvider(delay=0.1), limits=Limits(call_timeout=0.01))

    async def test_run_deadline(self):
        with self.assertRaisesRegex(RelayError, "deadline"):
            await self.run_demo(provider=SimulatedProvider(delay=0.02), limits=Limits(run_timeout=0.01))

    async def test_progress_failure_does_not_lose_answer(self):
        async def broken_emit(event):
            raise RuntimeError("Disconnected")
        with self.assertLogs("horizon_relay.engine", level="WARNING"):
            result = await RelayEngine(SimulatedProvider(), policy=PLANS).run("Goal", demo_sources(), emit=broken_emit)
        self.assertIn("SIMULATED", result.content)

    async def test_runs_have_distinct_ids_and_no_shared_results(self):
        engine = RelayEngine(SimulatedProvider(), policy=PLANS)
        a, b = await asyncio.gather(engine.run("A", demo_sources()), engine.run("B", demo_sources()))
        self.assertNotEqual(a.run_id, b.run_id)
        a.results.clear()
        self.assertEqual(len(b.results), 3)

    async def test_status_persistence_is_serialized(self):
        saved = []
        async def read_append_write(event):
            nonlocal saved
            snapshot = list(saved)
            await asyncio.sleep(0.001)
            saved = snapshot + [event.seq]
        await RelayEngine(SimulatedProvider("escalate"), Limits(local_concurrency=2), PLANS).run(
            "Goal", demo_sources(), emit=read_append_write,
        )
        self.assertGreater(len(saved), 18)
        self.assertEqual(saved, list(range(1, len(saved) + 1)))

    async def test_plan_validation(self):
        provider = SimulatedProvider()
        raw = (await provider.complete("cloud", "plan", {"sources": demo_sources()})).value
        import copy
        mutations = {
            "cycle": lambda p: p["tasks"][0]["depends_on"].append("group_reports"),
            "duplicate": lambda p: p["tasks"].append(copy.deepcopy(p["tasks"][0])),
            "unknown_source": lambda p: p["tasks"][0].update(input_refs=["sources.nope"]),
            "undeclared_result": lambda p: p["tasks"][0].update(input_refs=["results.extract_report_2"]),
            "omit_check": lambda p: p["tasks"][0].update(checks=[]),
            "unknown_check": lambda p: p["tasks"][0]["checks"].append("run_code"),
            "arbitrary_endpoint": lambda p: p["tasks"][0].update(endpoint="https://example.com"),
            "unknown_tier": lambda p: p["tasks"][0].update(preferred_tier="other"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                bad = copy.deepcopy(raw)
                mutate(bad)
                with self.assertRaises(ValidationError):
                    Plan.parse(bad, demo_sources(), 6)

    async def test_bad_group_ids_rejected(self):
        class BadGroup(SimulatedProvider):
            async def complete(self, tier, operation, payload):
                reply = await super().complete(tier, operation, payload)
                if operation == "work" and payload["task"]["id"] == "group_reports":
                    return replace(reply, value={"groups": [{"label": "Missing a report", "report_ids": ["report_1"]}]})
                return reply
        with self.assertRaises(ValidationError):
            await self.run_demo(provider=BadGroup())


if __name__ == "__main__":
    unittest.main()
