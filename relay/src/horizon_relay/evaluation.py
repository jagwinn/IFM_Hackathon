"""Routing tests: labeled requests with the route we expect, run through a Relay and scored.

The default set is fixtures/routing_tests.json. Each case has name, category, difficulty, prompt and
expected_route: the tiers that may answer it ("local" is the smallest model, then "mid", then "cloud"),
as a list or one string ("either" allows any). Optional: answer_pattern, a case-insensitive regular
expression the answer must match; policy, RoutingPolicy overrides for that case (to force a path);
expects, what the run itself must show (see check_expectations); note, why the case exists.
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
    expects: dict | None = None  # local_subtasks_at_least, cloud_subtasks_at_most, parallel_work
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
    local_subtasks: int = 0  # subtasks the cloud delegated back to local models
    cloud_subtasks: int = 0
    parallel_ms: int = 0  # how long local and cloud subtasks ran at the same time
    unmet: list = field(default_factory=list)  # expectations from the case that the run did not meet
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
                        c["prompt"], c.get("answer_pattern"), c.get("policy"), c.get("expects"), c.get("note", ""))
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
    local_subtasks, cloud_subtasks, parallel_ms = delegation(metrics)
    case_result = CaseResult(case, route, route in case.expected_route, round(time.monotonic() - start, 2), correct,
                             answer=result.content,
                             scores={v["tier"]: v["signals"]["combined_score"] for v in metrics.get("verdicts", [])},
                             local_calls=metrics["local_calls"], cloud_calls=metrics["cloud_calls"],
                             local_subtasks=local_subtasks, cloud_subtasks=cloud_subtasks, parallel_ms=parallel_ms,
                             graph=graph.to_dict())
    case_result.unmet = check_expectations(case, case_result)
    return case_result


def delegation(metrics: dict) -> tuple[int, int, int]:
    """(local subtasks, cloud subtasks, milliseconds where two or more subtasks ran at the same time)."""
    work = [c for c in metrics.get("calls", []) if c["operation"] == "work" and c.get("started_ms") is not None]
    edges = sorted([(c["started_ms"], 1) for c in work] + [(c["started_ms"] + (c["elapsed_ms"] or 0), -1) for c in work])
    running = concurrent = 0
    previous = edges[0][0] if edges else 0
    for moment, change in edges:
        if running > 1:
            concurrent += moment - previous
        running += change
        previous = moment
    tasks = {c["task_id"]: c["tier"] for c in work}
    local = sum(tier != "cloud" for tier in tasks.values())
    return local, len(tasks) - local, concurrent


def check_expectations(case: TestCase, result: "CaseResult") -> list[str]:
    """What the case demanded of the run itself and did not get."""
    unmet, expects = [], case.expects or {}
    least = expects.get("local_subtasks_at_least")
    if least is not None and result.local_subtasks < least:
        unmet.append(f"delegated {result.local_subtasks} subtasks to local models, expected at least {least}")
    most = expects.get("cloud_subtasks_at_most")
    if most is not None and result.cloud_subtasks > most:
        unmet.append(f"kept {result.cloud_subtasks} subtasks on the cloud, expected at most {most}")
    if expects.get("parallel_work") and result.parallel_ms <= 0:
        unmet.append("no two subtasks ever ran at the same time")
    return unmet


def summarize(results: list[CaseResult]) -> dict:
    done = [r for r in results if r.route != "error"]
    checked = [r for r in results if r.correct is not None]
    return {
        "cases": len(results),
        "matched": sum(r.matched for r in results),
        "answers_checked": len(checked),
        "answers_correct": sum(bool(r.correct) for r in checked),
        "expectations_checked": sum(bool(r.case.expects) for r in results),
        "expectations_met": sum(bool(r.case.expects) and not r.unmet for r in results),
        "delegated_subtasks": sum(r.local_subtasks for r in results),
        "errors": len(results) - len(done),
        "by_tier": {tier: sum(r.route == tier for r in done) for tier in TIERS},
        "cloud_calls": sum(r.cloud_calls for r in results),
        "local_calls": sum(r.local_calls for r in results),
        "seconds": round(sum(r.elapsed_s for r in results), 1),
    }
