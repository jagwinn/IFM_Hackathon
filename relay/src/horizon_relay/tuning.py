"""Tune the routing policy against the labeled routing tests.

1. profile(): every local model answers every test and is critiqued, with escalation forced and the cloud
   disabled, so it costs no cloud calls. The raw signals, critic verdict and answer check are recorded.
2. search(): replays those profiles under many candidate policies (weights, per-model thresholds, critic
   rule, the token rule) and ranks them: routes that match the labels count +1, a wrong local answer that
   would have been returned counts -2, and going to the cloud although a local model had already answered
   correctly costs 1.0 - that is the escalation with nothing to show for it. Climbing from the 0.9B to the 4B
   over a correct answer costs 0.1, the same as any cloud route: it spends a little local time, not credits.

With ~20 labeled tests this finds a sensible setting, not a statistically validated one; re-run it when the
models, prompts or tests change.
"""

from dataclasses import replace
import itertools
import re

from .errors import RelayError
from .evaluation import TestCase
from .policy import WEIGHTS, request_parts, task_complexity

SIGNAL_KEYS = tuple(WEIGHTS)


async def profile_case(relay, case: TestCase) -> dict:
    """What each local model answers for this case and every routing signal it produces."""
    probe = relay.__class__(
        relay.provider, limits=replace(relay.engine.limits, cloud_enabled=False),
        policy=replace(relay.engine.policy, rules=(), escalation_threshold=0.0, tier_thresholds={}, skip_small_above=None))
    verdicts = {}

    async def collect(event):
        if event.name == "local_verdict":
            verdicts[event.data["tier"]] = event.data["verdict"]

    try:
        await probe.chat([{"role": "user", "content": case.prompt}], emit=collect)
    except RelayError:
        pass  # expected: nothing local is trusted and the cloud is disabled
    tiers = {}
    for tier, v in verdicts.items():
        answer = v["samples"][0]["answer"]
        tiers[tier] = {
            "signals": {k: v["signals"][k] for k in SIGNAL_KEYS},
            "token_stats": {k: (v["samples"][0].get("token_stats") or {}).get(k) for k in ("prob", "hesitation", "count")},
            "answer_tokens": max(((s.get("token_stats") or {}).get("count") or 0) for s in v["samples"]),
            "requested": any(s["needs_escalation"] for s in v["samples"]),
            "empty": not answer.strip(),
            "critic": {k: v["critique"][k] for k in ("passed", "severity", "skipped")},
            "answer": answer[:500],
            "correct": bool(re.search(case.answer_pattern, answer, re.I)) if case.answer_pattern else None,
        }
    return {"name": case.name, "expected": list(case.expected_route), "complexity": task_complexity(case.prompt),
            "parts": request_parts(case.prompt), "tiers": tiers}


async def profile(relay, cases: list[TestCase], progress=None) -> list[dict]:
    results = []
    for i, case in enumerate(c for c in cases if not c.policy):
        if progress:
            progress(i, case)
        results.append(await profile_case(relay, case))
    return results


def _combiner(weights: dict):
    """RoutingPolicy.combine without building a policy: the weighted mean of the available signals."""
    def combine(signals: dict) -> float:
        total = score = 0.0
        for key, weight in weights.items():
            value = signals.get(key)
            if value is not None:
                total += weight
                score += weight * value
        return min(1.0, score / total) if total else 0.0
    return combine


def signals_of(params: dict, tier: dict) -> dict:
    """The tier's signals under this policy's token statistic ("hesitation" or "mean")."""
    signals = dict(tier["signals"])
    stats = tier.get("token_stats") or {}
    if params.get("token_signal") == "mean" and stats.get("prob") is not None:
        signals["token_uncertainty"] = 1.0 - stats["prob"]
    elif params.get("token_signal") == "mean":
        signals["token_uncertainty"] = None
    return signals


def simulate(params: dict, prof: dict, combine=None) -> str:
    """The tier this policy would route the profiled case to."""
    combine = combine or _combiner(params["weights"])
    if params.get("big_goal_parts") and prof.get("parts", 0) >= params["big_goal_parts"]:
        return "cloud"  # worth planning and handing out rather than answering locally
    local = [t for t in ("local", "mid") if t in prof["tiers"]]
    for i, tier in enumerate(local):
        if i == 0 and len(local) > 1 and params["skip_small_above"] is not None \
                and prof["complexity"] >= params["skip_small_above"]:
            continue
        t = prof["tiers"][tier]
        signals = signals_of(params, t)
        critic = t["critic"]
        if t["requested"] or t["empty"]:
            continue
        if params["critic_min"] is not None and not critic["skipped"] and not critic["passed"] \
                and critic["severity"] >= params["critic_min"]:
            continue
        token = signals.get("token_uncertainty")
        if (tier == "local" and params.get("token_max") is not None and token is not None
                and token >= params["token_max"] and t.get("answer_tokens", 0) >= params.get("token_min_length", 1)):
            continue
        if combine(signals) < params["thresholds"][tier]:
            return tier
    return "cloud"


def evaluate(params: dict, profiles: list[dict]) -> dict:
    matched = wrong = cloud = passed_over = wasted_cloud = 0
    routes = []
    combine = _combiner(params["weights"])
    order = ("local", "mid", "cloud")
    for prof in profiles:
        route = simulate(params, prof, combine)
        routes.append(route)
        matched += route in prof["expected"]
        cloud += route == "cloud"
        wrong += route != "cloud" and prof["tiers"][route]["correct"] is False
        # Escalating past a model that had it right buys nothing; paying the cloud for it is the waste that matters.
        passed_over += sum(1 for tier in order[:order.index(route)]
                           if prof["tiers"].get(tier, {}).get("correct") is True)
        wasted_cloud += route == "cloud" and any(prof["tiers"].get(tier, {}).get("correct") is True
                                                 for tier in ("local", "mid"))
    return {"objective": round(matched - 2 * wrong - wasted_cloud - 0.1 * (passed_over + cloud), 3),
            "matched": matched, "wrong_local": wrong, "passed_over_correct": passed_over,
            "wasted_cloud": wasted_cloud, "cloud": cloud, "cases": len(profiles), "routes": routes}


def search(profiles: list[dict]) -> list[tuple[dict, dict]]:
    """Candidate policies ranked best first (objective, then fewer cloud routes, then closer to the defaults)."""
    grid = {
        "self_uncertainty": (0.10, 0.25, 0.40), "token_uncertainty": (0.0, 0.15, 0.30),
        "inconsistency": (0.05, 0.15, 0.25), "critic_risk": (0.15, 0.25, 0.40),
        "task_complexity": (0.10, 0.25), "explicit_escalation": (0.10,),
    }
    thresholds = (0.20, 0.25, 0.30, 0.40, 0.50, 0.55)
    ranked = []
    for values in itertools.product(*grid.values()):
        weights = dict(zip(grid, values))
        for local_t, mid_t, critic_min, token_max, token_min_length, token_signal in itertools.product(
                thresholds, thresholds, (None, 0.5), (None, 0.10, 0.12, 0.15, 0.20), (1, 3), ("hesitation", "mean")):
            skip = None
            if not local_t <= mid_t + 0.1:
                continue  # the smallest model should not be trusted much more easily than the larger one
            params = {"weights": weights, "thresholds": {"local": local_t, "mid": mid_t}, "critic_min": critic_min,
                      "skip_small_above": skip, "token_max": token_max, "token_min_length": token_min_length,
                      "token_signal": token_signal}
            result = evaluate(params, profiles)
            distance = sum(abs(weights[k] - WEIGHTS[k]) for k in weights) + abs(local_t - 0.5) + abs(mid_t - 0.5)
            ranked.append((result["objective"], -result["cloud"], -distance, params, result))
    ranked.sort(key=lambda r: r[:3], reverse=True)
    return [(r[3], r[4]) for r in ranked[:50]]
