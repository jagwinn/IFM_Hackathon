"""Routing rules: when a local answer can be trusted, and what happens when it cannot.

Each local model, smallest first, answers the request itself and judges its own answer
(`local_samples` times), then a critic checks it: the next larger local model when there is one
(`cross_critic`), since a model rarely catches its own mistakes. Six signals are combined into a score:

    self_uncertainty     1 - the model's average self-reported confidence
    token_uncertainty    1 - the geometric-mean probability of the answer's own tokens (logprobs), when available
    inconsistency        how much its samples disagree
    critic_risk          how serious a flaw the critic found (0-1)
    task_complexity      a small keyword/length heuristic over the request
    explicit_escalation  1 if the model said it needs a larger model

The score is the weighted mean of the signals that are available (weights are relative).

Defaults were tuned with `horizon-relay tune` on the built-in routing tests (K2-Horizon 0.9B and 4B on an
RTX 3070): 20 of 22 tests routed as labeled and one wrong local answer returned, against 12 of 22 and six
wrong answers before tuning. Re-tune when models change.

`rules` run first and may force a decision; otherwise a score at or above the tier's threshold
escalates to the next model up. Requests whose task_complexity reaches `skip_small_above` skip the
smallest model entirely. Past the largest local model the cloud takes over: it plans and
delegates subtasks (cloud_mode="plan") or answers directly ("direct").

Tune by passing a RoutingPolicy to Relay, or with environment variables (see from_env).
"""

from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
import os
import re
from statistics import mean
from typing import Any, Callable, Mapping

from .errors import ValidationError
from .plan import Task
from .results import validate_result


@dataclass(frozen=True)
class Assessment:
    """One local answer and the model's own judgment of it."""
    answer: str
    confidence: float  # 0-1, self-reported
    task_type: str
    difficulty: str  # low, medium, high
    needs_escalation: bool
    reason: str
    structured: bool = True  # False when the model ignored the required JSON shape
    repaired: bool = False  # True when fields were recovered from slightly malformed JSON
    token_prob: float | None = None  # geometric-mean probability of the answer's tokens, if logprobs were returned
    token_stats: dict | None = None  # tokens.answer_token_stats(): per-token probabilities and the weakest token


@dataclass(frozen=True)
class Critique:
    passed: bool
    severity: float  # 0 = no issue, 1 = answer likely unusable
    critique: str
    skipped: bool = False
    tier: str = ""  # the model that critiqued


@dataclass(frozen=True)
class Signals:
    self_uncertainty: float
    token_uncertainty: float | None  # None when the model returned no token log-probabilities
    inconsistency: float
    critic_risk: float
    task_complexity: float
    explicit_escalation: float
    combined_score: float


@dataclass(frozen=True)
class Verdict:
    """Everything the policy saw for one local model, and what it decided."""
    tier: str
    samples: tuple[Assessment, ...]
    critique: Critique
    signals: Signals
    threshold: float
    escalate: bool
    reason: str  # the rule that decided, or the score comparison

    @property
    def answer(self) -> str:
        return self.samples[0].answer

    def to_dict(self) -> dict:
        return asdict(self)


Rule = Callable[[Verdict], bool | None]  # True: escalate, False: keep local, None: no opinion


def escalate_when_requested(verdict: Verdict) -> bool | None:
    """The model said it needs a larger model, or ignored the answer format: do not trust it."""
    if any(s.needs_escalation for s in verdict.samples):
        return True
    return None


def reject_empty_answers(verdict: Verdict) -> bool | None:
    return True if not verdict.answer.strip() else None


def token_uncertainty_at_least(limit: float, tiers: tuple[str, ...] = ("local",), min_tokens: int = 3) -> Rule:
    """Escalate when the model hesitated over the tokens of its own answer. On the smallest model this
    separates easy requests (token uncertainty up to ~0.07) from everything harder (~0.19 and up).

    Answers shorter than `min_tokens` are left to the score: in a one-word answer a single hesitant token
    ("positive" against "Positive") swings the whole measure without meaning the model is unsure."""
    def escalate_when_tokens_are_uncertain(verdict: Verdict) -> bool | None:
        value = verdict.signals.token_uncertainty
        tokens = max((s.token_stats or {}).get("count", 0) for s in verdict.samples)
        if verdict.tier in tiers and value is not None and value >= limit and tokens >= min_tokens:
            return True
        return None
    escalate_when_tokens_are_uncertain.limit = limit
    escalate_when_tokens_are_uncertain.min_tokens = min_tokens
    return escalate_when_tokens_are_uncertain


def critic_fails(min_severity: float = 0.5) -> Rule:
    """Escalate when the critic rejects the answer with at least this severity."""
    def escalate_when_critic_rejects(verdict: Verdict) -> bool | None:
        c = verdict.critique
        return True if not c.skipped and not c.passed and c.severity >= min_severity else None
    escalate_when_critic_rejects.min_severity = min_severity
    return escalate_when_critic_rejects


WEIGHTS = {
    "self_uncertainty": 0.25,
    "token_uncertainty": 0.15,
    "inconsistency": 0.25,
    "critic_risk": 0.25,
    "task_complexity": 0.10,
    "explicit_escalation": 0.10,
}

HARD_MARKERS = ("prove", "derive", "counterexample", "debug", "race condition", "algorithm", "probability",
                "bayes", "causal", "optimize", "complexity", "contradiction", "multi-step", "constraint",
                "ambiguous", "legal", "medical", "financial")
MEDIUM_MARKERS = ("compare", "explain why", "analyze", "reason", "tradeoff", "plan", "design", "evaluate",
                  "infer", "calculate")


def task_complexity(text: str) -> float:
    """Small deterministic signal; intentionally simple and easy to explain."""
    p = text.lower()
    score = 0.05 + 0.07 * sum(m in p for m in MEDIUM_MARKERS) + 0.12 * sum(m in p for m in HARD_MARKERS)
    score += 0.10 * (len(text) > 500) + 0.10 * (len(text) > 1200) + 0.08 * (text.count("?") > 2)
    return _clamp(score)


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, " ".join(a.lower().split()), " ".join(b.lower().split())).ratio()


def _salvage(text: str) -> dict | None:
    """Recover the solver's fields from nearly valid JSON, e.g. an unescaped quote inside "reason"."""
    answer = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    confidence = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', text)
    if not answer or not confidence:
        return None
    fields = {"answer": answer.group(1).replace('\\"', '"'), "confidence": confidence.group(1)}
    for key in ("task_type", "difficulty"):
        match = re.search(rf'"{key}"\s*:\s*"([^"]*)"', text)
        if match:
            fields[key] = match.group(1)
    escalate = re.search(r'"needs_escalation"\s*:\s*(true|false)', text)
    fields["needs_escalation"] = escalate is not None and escalate.group(1) == "true"
    reason = re.search(r'"reason"\s*:\s*"(.*)', text, re.S)
    fields["reason"] = reason.group(1).rstrip('"} \n') if reason else ""
    return fields


def read_assessment(value: Any, token_stats: dict | None = None) -> Assessment:
    """The solver's JSON; anything else is treated as an unreliable answer that needs escalation."""
    repaired = False
    if isinstance(value, str) and (salvaged := _salvage(value)):
        value, repaired = salvaged, True
    if isinstance(value, dict) and isinstance(value.get("answer"), (str, int, float)):
        try:
            confidence = _clamp(float(value.get("confidence", 0.5)))
        except (TypeError, ValueError):
            confidence = 0.5
        return Assessment(str(value["answer"]), confidence, str(value.get("task_type", "unknown"))[:80],
                          str(value.get("difficulty", "medium"))[:20], bool(value.get("needs_escalation", False)),
                          str(value.get("reason", ""))[:500], repaired=repaired,
                          token_prob=token_stats["prob"] if token_stats else None, token_stats=token_stats)
    return Assessment(str(value)[:4000], 0.15, "unstructured", "high", True,
                      "The model did not return the required JSON, so its answer is treated as unreliable.", False,
                      token_prob=token_stats["prob"] if token_stats else None, token_stats=token_stats)


def read_critique(value: Any) -> Critique:
    if isinstance(value, dict) and "severity" in value:
        try:
            severity = _clamp(float(value["severity"]))
        except (TypeError, ValueError):
            severity = 1.0
        return Critique(bool(value.get("passed", False)), severity, str(value.get("critique", ""))[:1000])
    return Critique(False, 1.0, "The critic did not return valid JSON, so verification counts as failed.")


def answer_exact_extraction(goal: str, sources: dict[str, str], candidate: Any) -> str | None:
    """For an opted-in verbatim copy of one source (/local-extract): accept only if the quote is the whole source."""
    if len(sources) != 1:
        return None
    inputs = {f"sources.{key}": text for key, text in sources.items()}
    try:
        validate_result("report_extraction", candidate, inputs)
    except ValidationError:
        return None
    if candidate["quote"] != next(iter(sources.values())):
        return None
    return f'**{candidate["report_id"]}**: {candidate["quote"]}'


@dataclass(frozen=True)
class RoutingPolicy:
    escalation_threshold: float = 0.50
    tier_thresholds: dict = field(default_factory=lambda: {"mid": 0.25})  # per-tier override of the threshold
    local_samples: int = 2  # answers per local model; their disagreement is the inconsistency signal
    use_critic: bool = True
    weights: dict = field(default_factory=lambda: dict(WEIGHTS))
    rules: tuple[Rule, ...] = (escalate_when_requested, reject_empty_answers, critic_fails(0.5),
                               token_uncertainty_at_least(0.12, tiers=("local",)))
    cross_critic: bool = True  # the next larger local model critiques; the largest critiques itself
    skip_small_above: float | None = None  # task_complexity at which the smallest model is skipped
    cloud_mode: str = "plan"  # "plan": plan and delegate subtasks back to local models; "direct": cloud answers
    # Plan mode only:
    local_worker_attempts: int = 2  # tries on the assigned local model before a subtask fails or escalates
    escalate_failed_local_tasks: bool = True  # then one try on each larger tier, for that subtask only
    plan_attempts: int = 2  # the planner gets its validation error back and may retry

    def __post_init__(self):
        if not all(0 <= t <= 1 for t in (self.escalation_threshold, *self.tier_thresholds.values())):
            raise ValueError("thresholds must be between 0 and 1")
        for name in ("local_samples", "local_worker_attempts", "plan_attempts"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.cloud_mode not in ("plan", "direct"):
            raise ValueError("cloud_mode must be 'plan' or 'direct'")
        if set(self.weights) != set(WEIGHTS):
            raise ValueError(f"weights must cover exactly {sorted(WEIGHTS)}")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "RoutingPolicy":
        """ESCALATION_THRESHOLD, LOCAL_SAMPLES, RELAY_USE_CRITIC, RELAY_CROSS_CRITIC, RELAY_CLOUD_MODE,
        SKIP_SMALL_ABOVE. Unset variables keep the defaults above."""
        env = os.environ if env is None else env
        defaults = cls()
        skip = env.get("SKIP_SMALL_ABOVE")
        return cls(escalation_threshold=float(env.get("ESCALATION_THRESHOLD", defaults.escalation_threshold)),
                   tier_thresholds=defaults.tier_thresholds,
                   local_samples=int(env.get("LOCAL_SAMPLES", defaults.local_samples)),
                   use_critic=env.get("RELAY_USE_CRITIC", "true").lower() == "true",
                   cross_critic=env.get("RELAY_CROSS_CRITIC", "true").lower() == "true",
                   cloud_mode=env.get("RELAY_CLOUD_MODE", defaults.cloud_mode),
                   skip_small_above=float(skip) if skip not in (None, "", "none") else defaults.skip_small_above)

    def critic_tier(self, tier: str, tiers: tuple[str, ...]) -> str:
        """Which model critiques `tier`'s answer: the next larger local model if cross_critic, else itself."""
        local = [t for t in tiers if t != "cloud"]
        if self.cross_critic and tier in local and local.index(tier) + 1 < len(local):
            return local[local.index(tier) + 1]
        return tier

    def skips(self, tier: str, tiers: tuple[str, ...], complexity: float) -> bool:
        """Skip the smallest local model for requests that look hard, when a larger local model exists."""
        local = [t for t in tiers if t != "cloud"]
        return (self.skip_small_above is not None and len(local) > 1 and tier == local[0]
                and complexity >= self.skip_small_above)

    def signals(self, text: str, samples: list[Assessment], critique: Critique) -> Signals:
        uncertainty = 1.0 - mean(s.confidence for s in samples)
        pairs = [(a, b) for i, a in enumerate(samples) for b in samples[i + 1:]]
        inconsistency = 1.0 - mean(similarity(a.answer, b.answer) for a, b in pairs) if pairs else 0.0
        token_probs = [s.token_prob for s in samples if s.token_prob is not None]
        values = {"self_uncertainty": uncertainty,
                  "token_uncertainty": 1.0 - mean(token_probs) if token_probs else None,
                  "inconsistency": inconsistency, "critic_risk": critique.severity,
                  "task_complexity": task_complexity(text),
                  "explicit_escalation": 1.0 if any(s.needs_escalation for s in samples) else 0.0}
        return Signals(**{k: None if v is None else round(v, 4) for k, v in values.items()},
                       combined_score=round(self.combine(values), 4))

    def combine(self, values: dict) -> float:
        """Weighted mean of the available signals (a signal of None is left out)."""
        available = {k: v for k, v in values.items() if v is not None and k in self.weights}
        total = sum(self.weights[k] for k in available)
        return _clamp(sum(self.weights[k] * v for k, v in available.items()) / total) if total else 0.0

    def threshold(self, tier: str) -> float:
        return self.tier_thresholds.get(tier, self.escalation_threshold)

    def decide(self, tier: str, text: str, samples: list[Assessment], critique: Critique) -> Verdict:
        signals = self.signals(text, samples, critique)
        threshold = self.threshold(tier)
        base = Verdict(tier, tuple(samples), critique, signals, threshold, False, "")
        for rule in self.rules:
            forced = rule(base)
            if forced is not None:
                name = getattr(rule, "__name__", "rule").replace("_", " ")
                if name == "escalate when critic rejects":
                    name += f" (severity {critique.severity:.2f})"
                if name == "escalate when tokens are uncertain":
                    name += f" ({signals.token_uncertainty:.2f} ≥ {rule.limit:.2f})"
                return Verdict(**{**base.__dict__, "escalate": forced, "reason": f"Rule: {name}"})
        escalate = signals.combined_score >= threshold
        comparison = "≥" if escalate else "<"
        return Verdict(**{**base.__dict__, "escalate": escalate,
                          "reason": f"Score {signals.combined_score:.2f} {comparison} threshold {threshold:.2f}"})

    def worker_tiers(self, task: Task, tiers: tuple[str, ...] = ("local", "cloud")) -> tuple[str, ...]:
        """Plan mode: the tier for each attempt at a subtask, in order. `tiers` lists available tiers, smallest first.

        With local, mid and cloud: a local task tries local, local, mid, cloud; a mid task mid, mid, cloud.
        """
        if task.preferred_tier == "cloud":
            return ("cloud",)
        larger = tiers[tiers.index(task.preferred_tier) + 1:]
        return (task.preferred_tier,) * self.local_worker_attempts + (larger if self.escalate_failed_local_tasks else ())


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
