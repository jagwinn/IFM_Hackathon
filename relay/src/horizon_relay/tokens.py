"""Token log-probabilities: how sure a model was of the exact tokens it produced.

Providers that support it attach tokens to a Reply as [{"t": token, "lp": logprob, "top": [[token, logprob], ...]}].
answer_token_stats() finds the tokens of one JSON field's value (by default "answer") and summarizes them.
"""

import math
import re

MAX_TOKENS_SHOWN = 120


def _value_span(text: str, field: str) -> tuple[int, int] | None:
    """Character span of a JSON field's value, without its quotes, or None."""
    match = re.search(rf'"{field}"\s*:\s*', text)
    if not match:
        return None
    start = match.end()
    if start < len(text) and text[start] == '"':
        start += 1
        end = start
        while end < len(text) and text[end] != '"':
            end += 2 if text[end] == "\\" else 1
        return start, min(end, len(text))
    end = start
    while end < len(text) and text[end] not in ",}\n":
        end += 1
    return start, end


def answer_token_stats(tokens: list[dict] | None, field: str = "answer") -> dict | None:
    """Summary of the tokens that spell the field's value; the whole output if the field cannot be found."""
    if not tokens:
        return None
    text = "".join(t["t"] for t in tokens)
    span = _value_span(text, field)
    offsets, position = [], 0
    for token in tokens:
        offsets.append((position, position + len(token["t"])))
        position += len(token["t"])
    if span:
        chosen = [t for t, (a, b) in zip(tokens, offsets) if a < span[1] and b > span[0] and t["t"].strip()]
    else:
        chosen = [t for t in tokens if t["t"].strip()]
    if not chosen:
        return None
    logprobs = [t["lp"] for t in chosen]
    mean_lp = sum(logprobs) / len(logprobs)
    weakest = min(chosen, key=lambda t: t["lp"])
    return {
        "span": field if span else "output",
        "count": len(chosen),
        "mean_logprob": round(mean_lp, 4),
        "prob": round(math.exp(mean_lp), 4),  # geometric-mean probability per token
        "min_prob": round(math.exp(weakest["lp"]), 4),
        "min_token": weakest["t"],
        "tokens": [{"t": t["t"], "p": round(math.exp(t["lp"]), 4),
                    "alt": [[a, round(math.exp(lp), 4)] for a, lp in t.get("top", [])[:3]]}
                   for t in chosen[:MAX_TOKENS_SHOWN]],
    }
