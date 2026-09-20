import unittest

from horizon_relay import Limits, Relay, RoutingPolicy, SimulatedProvider
from horizon_relay.evaluation import CaseResult, TestCase, load_cases, run_case, select_cases, summarize

from test_modular import SmallUnsureMidSure


class EvaluationTests(unittest.IsolatedAsyncioTestCase):
    def test_builtin_cases_cover_every_tier(self):
        cases = load_cases()
        for tier in ("local", "mid", "cloud"):
            with self.subTest(tier=tier):
                self.assertGreaterEqual(sum(c.expected_route == (tier,) for c in cases), 5)
        self.assertGreaterEqual(len(select_cases(cases, "forced")), 3)
        self.assertEqual(select_cases(cases, "4b"), select_cases(cases, "mid"))
        self.assertEqual([c.name for c in select_cases(cases, "1, 3")], [cases[0].name, cases[2].name])
        self.assertTrue(all(c.difficulty == "hard" for c in select_cases(cases, "hard")))

    def test_legacy_expected_routes(self):
        import json, tempfile, os
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump([{"name": "a", "category": "c", "difficulty": "easy", "expected_route": "either", "prompt": "p"},
                       {"name": "b", "category": "c", "difficulty": "easy", "expected_route": "local", "prompt": "p"}], f)
        self.addCleanup(os.remove, f.name)
        a, b = load_cases(f.name)
        self.assertEqual((a.expected_route, b.expected_route), (("local", "mid", "cloud"), ("local",)))

    async def test_run_case_records_route_answer_and_graph(self):
        case = TestCase("t", "c", "easy", ("local",), "Summarize report 1", answer_pattern=r"checkout")
        local = await run_case(Relay(SimulatedProvider("trusted")), case)
        self.assertEqual((local.route, local.matched, local.correct, local.cloud_calls), ("local", True, True, 0))
        self.assertEqual(local.graph["route"], "local")
        cloud = await run_case(Relay(SimulatedProvider(), policy=RoutingPolicy(cloud_mode="direct")), case)
        self.assertEqual((cloud.route, cloud.matched, cloud.correct), ("cloud", False, True))
        stopped = await run_case(Relay(SimulatedProvider(), limits=Limits(cloud_enabled=False)), case)
        self.assertEqual(stopped.route, "error")
        summary = summarize([local, cloud, stopped])
        self.assertEqual((summary["matched"], summary["errors"], summary["answers_correct"], summary["answers_checked"]),
                         (1, 1, 2, 3))

    async def test_delegation_expectations(self):
        from horizon_relay.evaluation import check_expectations, delegation
        metrics = {"calls": [
            {"operation": "work", "tier": "local", "task_id": "write_parse", "started_ms": 100, "elapsed_ms": 400},
            {"operation": "work", "tier": "mid", "task_id": "write_format", "started_ms": 120, "elapsed_ms": 500},
            {"operation": "work", "tier": "cloud", "task_id": "design_api", "started_ms": 150, "elapsed_ms": 900},
            {"operation": "plan", "tier": "cloud", "task_id": None, "started_ms": 0, "elapsed_ms": 90}]}
        local, cloud, parallel = delegation(metrics)
        self.assertEqual((local, cloud), (2, 1))
        self.assertEqual(parallel, 500)  # 120ms to 620ms had two or more subtasks in flight
        case = TestCase("t", "delegation", "hard", ("cloud",), "p",
                        expects={"local_subtasks_at_least": 2, "cloud_subtasks_at_most": 1, "parallel_work": True})
        result = CaseResult(case, "cloud", True, 1.0, local_subtasks=local, cloud_subtasks=cloud, parallel_ms=parallel)
        self.assertEqual(check_expectations(case, result), [])
        lonely = CaseResult(case, "cloud", True, 1.0, local_subtasks=0, cloud_subtasks=2, parallel_ms=0)
        self.assertEqual(len(check_expectations(case, lonely)), 3)

    async def test_case_policy_forces_the_path(self):
        forced_mid = TestCase("m", "forced", "easy", ("mid",), "Capital?", policy={"tier_thresholds": {"local": 0.0}})
        relay = Relay(SmallUnsureMidSure())
        self.assertEqual((await run_case(relay, forced_mid)).route, "mid")
        forced_cloud = TestCase("c", "forced", "hard", ("cloud",), "Capital?",
                                policy={"escalation_threshold": 0.0, "tier_thresholds": {}, "cloud_mode": "direct"})
        result = await run_case(relay, forced_cloud)
        self.assertEqual((result.route, result.cloud_calls), ("cloud", 1))
        self.assertEqual(relay.engine.policy, RoutingPolicy())  # the shared relay is unchanged
