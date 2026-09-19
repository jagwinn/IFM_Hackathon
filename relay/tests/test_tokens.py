import math
import unittest

from horizon_relay.tokens import answer_token_stats


def toks(*pairs):
    return [{"t": t, "lp": math.log(p), "top": [[t, math.log(p)], ["x", math.log(0.01)]]} for t, p in pairs]


class TokenTests(unittest.TestCase):
    def test_answer_string_tokens_only(self):
        tokens = toks(("</ifm|think_faster>", 0.9), ('{"', 1.0), ("answer", 1.0), ('":', 1.0), (' "', 0.8),
                      ("39", 0.5), ("1", 0.9), ('",', 1.0), (' "', 1.0), ("confidence", 1.0), ('":', 1.0), (" 1", 0.7), ("}", 1.0))
        stats = answer_token_stats(tokens)
        self.assertEqual(stats["span"], "answer")
        self.assertEqual([t["t"] for t in stats["tokens"]], ["39", "1"])
        self.assertAlmostEqual(stats["prob"], math.sqrt(0.5 * 0.9), places=3)
        self.assertEqual((stats["min_token"], stats["min_prob"]), ("39", 0.5))
        self.assertEqual(stats["tokens"][0]["alt"][1], ["x", 0.01])

    def test_numeric_answer_and_fallback(self):
        numeric = answer_token_stats(toks(('{"answer":', 1.0), (" ", 1.0), ("391", 0.6), (",", 1.0)))
        self.assertEqual([t["t"] for t in numeric["tokens"]], ["391"])
        free = answer_token_stats(toks(("Paris", 0.5), (".", 1.0)))
        self.assertEqual(free["span"], "output")
        self.assertIsNone(answer_token_stats(None))
