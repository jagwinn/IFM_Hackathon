"""Routing tests: labeled requests with the route we expect, run through a Relay and scored.

The default set is fixtures/routing_tests.json. Each case has name, category, difficulty, prompt and
expected_route: the tiers that may answer it ("local" is the smallest model, then "mid", then "cloud"),
as a list or one string ("either" allows any). Optional: answer_pattern, a case-insensitive regular
expression the answer must match; policy, RoutingPolicy overrides for that case (to force a path);
note, why the case exists.
"""

from dataclasses import asdict, dataclass, field
from importlib.resources import files
import json
from pathlib import Path
import re
import time

from .errors import RelayError
from .graph import RunGraph

TIERS = ("local", "mid", "cloud")
ALIASES = {"0.9b": "local", "small": "local", "4b": "mid", "375b": "cloud", "large": "cloud"}


@dataclass(frozen=True)
class TestCase:
    name: str
    category: str
    difficulty: str
    expected_route: tuple[str, ...]
    prompt: str
    answer_pattern: str | None = None
    policy: dict | None = None
    note: str = ""


@dataclass
class CaseResult:
    case: TestCase
    route: str  # the tier that answered ("local", "mid", "cloud") or "error"
    matched: bool  # routed to one of the expected tiers
    elapsed_s: float
    correct: bool | None = None  # answer matched answer_pattern; None when the case has no pattern
    answer: str = ""
    scores: dict = field(default_factory=dict)  # tier -> combined routing score
    local_calls: int = 0
    cloud_calls: int = 0
    error: str = ""
    graph: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _expected(value) -> tuple[str, ...]:
    if value == "either":
        return TIERS
    tiers = (value,) if isinstance(value, str) else tuple(value)
    if not tiers or any(t not in TIERS for t in tiers):
        raise ValueError(f"expected_route must use {TIERS} or 'either', got {value!r}")
    return tiers


def load_cases(path: str | Path | None = None) -> list[TestCase]:
    text = Path(path).read_text() if path else files("horizon_relay").joinpath("fixtures/routing_tests.json").read_text()
    cases = []
    for c in json.loads(text):
        case = TestCase(c["name"], c.get("category", ""), c.get("difficulty", ""), _expected(c["expected_route"]),
                        c["prompt"], c.get("answer_pattern"), c.get("policy"), c.get("note", ""))
        if case.answer_pattern:
            re.compile(case.answer_pattern)
        cases.append(case)
    return cases


def select_cases(cases: list[TestCase], query: str = "") -> list[TestCase]:
    """"" or "all": every case. "1 3 5" or "1,3": by position. "local"/"0.9b", "mid"/"4b", "cloud": cases
    expected on that tier. "forced": policy-forced paths. Otherwise a difficulty, category or name words."""
    query = query.strip().lower().replace(",", " ")
    if not query or query == "all":
        return list(cases)
    words = query.split()
    if all(w.isdigit() for w in words):
        return [cases[int(w) - 1] for w in words if 1 <= int(w) <= len(cases)]
    tier = ALIASES.get(query, query)
    if tier in TIERS:
        return [c for c in cases if tier in c.expected_route]
    if query == "forced":
        return [c for c in cases if c.policy]
    return [c for c in cases if query in (c.difficulty, c.category) or query in c.name.lower()]


async def run_case(relay, case: TestCase, emit=None) -> CaseResult:
    """Run one case (with its policy overrides). `emit` also receives the run's events, e.g. to draw its graph."""
    if case.policy:
        relay = relay.with_policy(**case.policy)
    graph = RunGraph()

    async def on_event(event):
        graph.add(event)
        if emit:
            await emit(event)

    start = time.monotonic()
    try:
        result = await relay.chat([{"role": "user", "content": case.prompt}], emit=on_event)
    except RelayError as exc:
        return CaseResult(case, "error", False, round(time.monotonic() - start, 2),
                          correct=False if case.answer_pattern else None, error=str(exc), graph=graph.to_dict())
    metrics = result.metrics
    route = metrics.get("route") or "cloud"
    correct = bool(re.search(case.answer_pattern, result.content, re.I)) if case.answer_pattern else None
    return CaseResult(case, route, route in case.expected_route, round(time.monotonic() - start, 2), correct,
                      answer=result.content,
                      scores={v["tier"]: v["signals"]["combined_score"] for v in metrics.get("verdicts", [])},
                      local_calls=metrics["local_calls"], cloud_calls=metrics["cloud_calls"], graph=graph.to_dict())


def summarize(results: list[CaseResult]) -> dict:
    done = [r for r in results if r.route != "error"]
    checked = [r for r in results if r.correct is not None]
    return {
        "cases": len(results),
        "matched": sum(r.matched for r in results),
        "answers_checked": len(checked),
        "answers_correct": sum(bool(r.correct) for r in checked),
        "errors": len(results) - len(done),
        "by_tier": {tier: sum(r.route == tier for r in done) for tier in TIERS},
        "cloud_calls": sum(r.cloud_calls for r in results),
        "local_calls": sum(r.local_calls for r in results),
        "seconds": round(sum(r.elapsed_s for r in results), 1),
    }
