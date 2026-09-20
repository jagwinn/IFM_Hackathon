"""The relay's public API and extension points, independent of any front end."""

import contextlib
import io
import json
from pathlib import Path
import re
import unittest

from horizon_relay import (Assessment, Critique, Relay, RelayError, RelaySettings, Reply, RoutingPolicy,
                           SimulatedProvider, ValidationError)
from horizon_relay.cli import main
from horizon_relay.demo import run_demo
from horizon_relay.policy import read_assessment, read_critique, task_complexity
from horizon_relay.providers.simulated import demo_sources

PACKAGE = Path(__file__).resolve().parents[1] / "src/horizon_relay"


class TextExtractor(SimulatedProvider):
    async def complete(self, tier, operation, payload):
        if operation == "attempt":
            return Reply({"report_id": "text", "quote": payload["sources"]["text"]}, "fake")
        return await super().complete(tier, operation, payload)


class SmallUnsureMidSure(SimulatedProvider):
    """The 0.9B is unsure; the 4B is confident and its critic agrees."""
    models = {"local": "small", "mid": "medium", "cloud": "large"}

    async def complete(self, tier, operation, payload):
        if operation == "solve":
            confident = tier == "mid"
            return Reply({"answer": "Paris", "confidence": 0.9 if confident else 0.2, "task_type": "fact",
                          "difficulty": "low", "needs_escalation": not confident, "reason": "r"}, tier)
        if operation == "critique":
            return Reply({"passed": True, "severity": 0.0, "critique": "Correct"}, tier)
        return await super().complete(tier, operation, payload)


class PolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_trusted_local_answer_stays_local(self):
        result = await run_demo("trusted")
        self.assertEqual((result.metrics["route"], result.metrics["cloud_calls"]), ("local", 0))
        verdict = result.metrics["verdicts"][0]
        self.assertFalse(verdict["escalate"])
        self.assertLess(verdict["signals"]["combined_score"], verdict["threshold"])
        self.assertEqual(len(verdict["samples"]), 2)  # two samples, then the critic
        self.assertFalse(verdict["critique"]["skipped"])

    async def test_explicit_escalation_skips_critic_and_goes_to_cloud(self):
        result = await run_demo("standard")
        verdict = result.metrics["verdicts"][0]
        self.assertEqual(verdict["reason"], "Rule: escalate when requested")
        self.assertTrue(verdict["critique"]["skipped"])
        self.assertEqual(len(verdict["samples"]), 1)
        self.assertEqual(result.metrics["route"], "cloud")

    async def test_threshold_controls_escalation(self):
        relay = Relay(SimulatedProvider("trusted"), policy=RoutingPolicy(escalation_threshold=0.0))
        result = await relay.run("Summarize report 1", demo_sources())
        self.assertEqual(result.metrics["route"], "cloud")

    async def test_tier_thresholds(self):
        relay = Relay(SmallUnsureMidSure(), policy=RoutingPolicy(rules=(), tier_thresholds={"mid": 0.0}))
        result = await relay.chat([{"role": "user", "content": "Capital of France?"}])
        self.assertEqual([v["threshold"] for v in result.metrics["verdicts"]], [0.5, 0.0])
        self.assertEqual(result.metrics["route"], "cloud")

    async def test_custom_rule_can_keep_work_local(self):
        policy = RoutingPolicy(rules=(lambda verdict: False,), cloud_mode="plan")
        result = await Relay(SimulatedProvider(), policy=policy).run("Prioritize", demo_sources())
        self.assertEqual((result.metrics["route"], result.metrics["cloud_calls"]), ("local", 0))

    async def test_cascade_climbs_to_the_mid_model(self):
        result = await Relay(SmallUnsureMidSure()).chat([{"role": "user", "content": "Capital of France?"}])
        self.assertEqual(result.metrics["route"], "mid")
        self.assertEqual([v["tier"] for v in result.metrics["verdicts"]], ["local", "mid"])
        self.assertEqual(result.metrics["cloud_calls"], 0)
        self.assertTrue(result.content.endswith("Paris"))

    async def test_failed_local_call_escalates_instead_of_stopping(self):
        class Truncated(SmallUnsureMidSure):
            async def complete(self, tier, operation, payload):
                if tier == "local" and operation == "solve":
                    raise RelayError("Local response was empty or reached its token limit")
                return await super().complete(tier, operation, payload)
        result = await Relay(Truncated()).chat([{"role": "user", "content": "Hard puzzle"}])
        self.assertEqual(result.metrics["route"], "mid")
        local = result.metrics["verdicts"][0]
        self.assertTrue(local["escalate"])
        self.assertIn("token limit", local["samples"][0]["reason"])

    async def test_larger_model_critiques_and_a_rejection_escalates(self):
        critics = []
        class Reviewed(SmallUnsureMidSure):
            async def complete(self, tier, operation, payload):
                if operation == "solve":
                    return Reply({"answer": "Lyon", "confidence": 0.95, "task_type": "fact", "difficulty": "low",
                                  "needs_escalation": False, "reason": "sure"}, tier)
                if operation == "critique":
                    critics.append((payload["for_tier"], tier))
                    wrong = payload["for_tier"] == "local"
                    return Reply({"passed": not wrong, "severity": 0.9 if wrong else 0.0, "critique": "c"}, tier)
                return await super().complete(tier, operation, payload)
        result = await Relay(Reviewed()).chat([{"role": "user", "content": "Capital of France?"}])
        self.assertEqual(critics, [("local", "mid"), ("mid", "mid")])  # the 4B checks the 0.9B, then itself
        self.assertEqual(result.metrics["verdicts"][0]["reason"], "Rule: escalate when critic rejects (severity 0.90)")
        self.assertEqual(result.metrics["route"], "mid")

    async def test_hard_requests_skip_the_smallest_model(self):
        relay = Relay(SmallUnsureMidSure(), policy=RoutingPolicy(skip_small_above=0.2))
        result = await relay.chat([{"role": "user", "content": "Derive the probability and prove it"}])
        self.assertEqual((result.metrics["skipped"], result.metrics["route"]), (["local"], "mid"))
        easy = await relay.chat([{"role": "user", "content": "Capital of France?"}])
        self.assertEqual(easy.metrics["skipped"], [])

    async def test_token_rule_ignores_very_short_answers(self):
        from horizon_relay.policy import Signals, Verdict, token_uncertainty_at_least
        rule = token_uncertainty_at_least(0.12, min_tokens=3)
        def verdict(tokens):
            sample = Assessment("positive", 0.95, "t", "low", False, "", token_prob=0.7,
                                token_stats={"count": tokens, "prob": 0.7})
            signals = Signals(0.05, 0.3, 0.0, 0.0, 0.05, 0.0, 0.1)
            return Verdict("local", (sample,), Critique(True, 0.0, ""), signals, 0.5, False, "")
        self.assertIsNone(rule(verdict(1)))  # one hesitant token is not doubt about the answer
        self.assertTrue(rule(verdict(6)))

    def test_planning_only_pays_off_for_several_parts(self):
        policy = RoutingPolicy()
        campus = ("Plan a two-day prototype for a campus lost-and-found app. Compare two storage choices, suggest a "
                  "minimal feature set, and identify the biggest implementation risks.")
        puzzle = ("You have 12 visually identical balls, one of a different weight. Using a balance scale exactly "
                  "three times, give a strategy that always identifies the odd ball and whether it is heavier.")
        self.assertTrue(policy.plans(campus))  # four separable deliverables
        self.assertFalse(policy.plans(puzzle))  # one chain of reasoning
        self.assertFalse(policy.plans("What is 17 times 23?"))
        self.assertTrue(RoutingPolicy(cloud_mode="plan").plans("What is 17 times 23?"))
        self.assertFalse(RoutingPolicy(cloud_mode="direct").plans(campus))

    async def test_short_requests_are_answered_not_split(self):
        calls = []
        class Recorder(SimulatedProvider):
            async def complete(self, tier, operation, payload):
                calls.append(operation)
                return await super().complete(tier, operation, payload)
        result = await Relay(Recorder()).chat([{"role": "user", "content": "Prove that 17 times 23 is 391."}])
        self.assertEqual(calls, ["solve", "answer"])  # no plan, no subtasks, one cloud call
        self.assertEqual(result.metrics["cloud_calls"], 1)

    async def test_direct_cloud_mode(self):
        calls = []
        class Recorder(SimulatedProvider):
            async def complete(self, tier, operation, payload):
                calls.append(operation)
                return await super().complete(tier, operation, payload)
        result = await Relay(Recorder(), policy=RoutingPolicy(cloud_mode="direct")).run("Prioritize", demo_sources())
        self.assertEqual(calls, ["solve", "answer"])
        self.assertIn("Direct cloud answer", result.content)

    def test_signal_math(self):
        policy = RoutingPolicy()
        samples = [Assessment("Detroit", 0.9, "t", "low", False, ""), Assessment("detroit", 0.7, "t", "low", False, "")]
        signals = policy.signals("Extract the city", samples, Critique(True, 0.2, ""))
        self.assertAlmostEqual(signals.self_uncertainty, 0.2)
        self.assertEqual(signals.inconsistency, 0.0)
        self.assertIsNone(signals.token_uncertainty)
        w = policy.weights  # no token log-probabilities: the other weights are rescaled
        expected = (w["self_uncertainty"] * 0.2 + w["critic_risk"] * 0.2 + w["task_complexity"] * signals.task_complexity) \
            / (sum(w.values()) - w["token_uncertainty"])
        self.assertAlmostEqual(signals.combined_score, expected, places=3)
        with_tokens = [Assessment("391", 0.9, "t", "low", False, "", token_prob=0.5)] * 2
        self.assertAlmostEqual(policy.signals("x", with_tokens, Critique(True, 0.0, "")).token_uncertainty, 0.5)

    def test_unstructured_outputs_count_against_the_answer(self):
        self.assertTrue(read_assessment("free text").needs_escalation)
        self.assertEqual(read_assessment("free text").confidence, 0.15)
        self.assertEqual(read_critique("oops").severity, 1.0)

    def test_nearly_valid_json_is_repaired(self):
        raw = ('{"answer": "Detroit", "confidence": 0.95, "task_type": "extraction", "difficulty": "low", '
               '"needs_escalation": false, "reason": "Clear city name after "in Detroit, Michigan."}')
        sample = read_assessment(raw)
        self.assertEqual((sample.answer, sample.confidence, sample.needs_escalation), ("Detroit", 0.95, False))
        self.assertTrue(sample.repaired and sample.structured)
        self.assertIn("Clear city name", sample.reason)

    def test_complexity_heuristic(self):
        self.assertLess(task_complexity("Extract the city"), task_complexity("Derive the probability and prove it"))

    def test_policy_from_env(self):
        policy = RoutingPolicy.from_env({"ESCALATION_THRESHOLD": "0.4", "LOCAL_SAMPLES": "1", "RELAY_CLOUD_MODE": "direct"})
        self.assertEqual((policy.escalation_threshold, policy.local_samples, policy.cloud_mode), (0.4, 1, "direct"))

    async def test_fewer_local_attempts_escalate_sooner(self):
        relay = Relay(SimulatedProvider("repair"), policy=RoutingPolicy(local_worker_attempts=1, cloud_mode="plan"))
        result = await relay.run("Prioritize", demo_sources())
        workers = [c for c in result.metrics["calls"] if c["operation"] == "work" and c["tier"] == "cloud"]
        self.assertEqual([c["task_id"] for c in workers], ["extract_report_1"])

    async def test_escalation_can_be_disabled(self):
        relay = Relay(SimulatedProvider("escalate"), policy=RoutingPolicy(escalate_failed_local_tasks=False, cloud_mode="plan"))
        with self.assertRaises(ValidationError):
            await relay.run("Prioritize", demo_sources())

    async def test_plan_attempts(self):
        calls = []
        class BadPlanner(SimulatedProvider):
            async def complete(self, tier, operation, payload):
                calls.append(operation)
                if operation == "plan":
                    return Reply({}, "fake")
                return await super().complete(tier, operation, payload)
        with self.assertRaises(ValidationError):
            await Relay(BadPlanner(), policy=RoutingPolicy(plan_attempts=1, cloud_mode="plan")).run("Goal", {"a": "text"})
        self.assertEqual(calls, ["solve", "plan"])

    def test_invalid_policy(self):
        for bad in ({"local_worker_attempts": 0}, {"escalation_threshold": 2}, {"cloud_mode": "other"}, {"weights": {}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                RoutingPolicy(**bad)


class RelayApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_local_extract_command(self):
        result = await Relay(TextExtractor()).chat([{"role": "user", "content": "/local-extract Due Friday."}])
        self.assertEqual(result.metrics["cloud_calls"], 0)
        self.assertIn("**text**: Due Friday.", result.content)

    async def test_chat_uses_whole_conversation_as_source(self):
        seen = []
        class Recorder(SimulatedProvider):
            async def complete(self, tier, operation, payload):
                seen.append(payload)
                return await super().complete(tier, operation, payload)
        await Relay(Recorder(), policy=RoutingPolicy(cloud_mode="plan")).chat(
            [{"role": "system", "content": "Be terse"}, {"role": "user", "content": "Go"}])
        self.assertEqual(seen[0]["task"], "system: Be terse\n\nuser: Go\n\nRespond to the last user message.")
        plan = next(p for p in seen if "validation_error" in p)
        self.assertEqual(plan["goal"], "Go")
        self.assertEqual(plan["sources"], {"conversation": "system: Be terse\n\nuser: Go"})

    async def test_chat_rejects_non_text_and_oversized_conversations(self):
        relay = Relay(SimulatedProvider())
        for messages in ([], [{"role": "tool", "content": "x"}], [{"role": "user", "content": "x" * 30000}]):
            with self.subTest(messages=str(messages)[:40]):
                with self.assertRaises(RelayError):
                    await relay.chat(messages)

    async def test_complete_is_a_single_local_call(self):
        self.assertIn("no model called", await Relay(SimulatedProvider()).complete([{"role": "user", "content": "Title?"}]))

    async def test_summary(self):
        result = await run_demo("local")
        self.assertEqual(result.summary(), "Relay: 1 local calls · 0 cloud calls · tokens: unavailable (simulation).")

    def test_settings_from_env(self):
        settings = RelaySettings.from_env({"CLOUD_API_KEY": "secret-key", "CLOUD_ENABLED": "false", "LOCAL_MODEL": "m"})
        self.assertEqual(settings.local.model, "m")
        self.assertFalse(settings.limits.cloud_enabled)
        self.assertEqual(settings.cloud.key, "secret-key")
        self.assertNotIn("secret-key", repr(settings))

    def test_cli_demo(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main(["demo", "--scenario", "local", "--json"])
        result = json.loads(out.getvalue())
        self.assertEqual(result["metrics"]["cloud_calls"], 0)
        self.assertIn("SIMULATED", result["content"])


class BoundaryTests(unittest.TestCase):
    def test_core_has_no_front_end_code(self):
        for path in PACKAGE.rglob("*.py"):
            with self.subTest(path=path.name):
                self.assertIsNone(re.search(r"open.?webui|__event_emitter__", path.read_text(), re.I))


if __name__ == "__main__":
    unittest.main()
