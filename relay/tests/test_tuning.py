import unittest

from horizon_relay import Relay
from horizon_relay.evaluation import TestCase
from horizon_relay.policy import WEIGHTS
from horizon_relay.tuning import evaluate, profile_case, search, simulate

from test_modular import SmallUnsureMidSure


def tier(score_signals, correct, requested=False, critic=(True, 0.0, False)):
    return {"signals": score_signals, "requested": requested, "empty": False, "correct": correct, "answer": "a",
            "critic": {"passed": critic[0], "severity": critic[1], "skipped": critic[2]}}


LOW = {k: 0.0 for k in WEIGHTS}
HIGH = {k: 1.0 for k in WEIGHTS}
PARAMS = {"weights": dict(WEIGHTS), "thresholds": {"local": 0.5, "mid": 0.5}, "critic_min": 0.5, "skip_small_above": None}


class TuningTests(unittest.IsolatedAsyncioTestCase):
    async def test_profile_records_every_local_model_without_the_cloud(self):
        prof = await profile_case(Relay(SmallUnsureMidSure()), TestCase("t", "c", "easy", ("mid",), "Capital?", r"paris"))
        self.assertEqual(set(prof["tiers"]), {"local", "mid"})
        self.assertTrue(prof["tiers"]["local"]["requested"])
        self.assertTrue(prof["tiers"]["mid"]["correct"])

    def test_simulate_follows_rules_thresholds_and_skip(self):
        prof = {"name": "x", "expected": ["mid"], "complexity": 0.5,
                "tiers": {"local": tier(LOW, False, critic=(False, 0.8, False)), "mid": tier(LOW, True)}}
        self.assertEqual(simulate(PARAMS, prof), "mid")  # the critic rule rejects the 0.9B
        self.assertEqual(simulate({**PARAMS, "critic_min": None}, prof), "local")
        self.assertEqual(simulate({**PARAMS, "critic_min": None, "skip_small_above": 0.4}, prof), "mid")
        prof["tiers"]["mid"] = tier(HIGH, True)
        self.assertEqual(simulate(PARAMS, prof), "cloud")

    def test_search_prefers_settings_that_avoid_wrong_local_answers(self):
        wrong_small = {"name": "a", "expected": ["mid"], "complexity": 0.05,
                       "tiers": {"local": tier({**LOW, "token_uncertainty": 0.6}, False), "mid": tier(LOW, True)}}
        easy = {"name": "b", "expected": ["local"], "complexity": 0.05,
                "tiers": {"local": tier(LOW, True), "mid": tier(LOW, True)}}
        self.assertEqual(evaluate(PARAMS, [wrong_small])["wrong_local"], 1)
        best_params, best = search([wrong_small, easy])[0]
        self.assertEqual((best["routes"], best["wrong_local"]), (["mid", "local"], 0))
