"""Model providers. The engine only needs `complete(tier, operation, payload) -> Reply`.

tier: "local", "mid" (optional) or "cloud", smallest to largest. operation: "attempt", "plan", "work", "synthesize" or "direct" (see prompts.py).
Implementations: openai_compatible.OpenAICompatibleProvider (real HTTP, needs httpx) and
simulated.SimulatedProvider (scripted, no network).
"""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Reply:
    value: Any  # parsed JSON for attempt/plan/work when possible, otherwise text
    model: str
    # None is deliberately different from measured zero tokens.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    # Per-token log-probabilities when the provider supports them (see tokens.py); None otherwise.
    tokens: list | None = None
    thinking: str = ""  # what the model reasoned before answering, when it exposes that
    text: str = ""  # the answer as text, before any JSON parsing


TIER_ORDER = ("local", "mid", "cloud")


class Provider(Protocol):
    simulated: bool  # scripted providers must say so; answers and events are then labeled
    models: dict[str, str]  # tier -> model name, for the tiers this provider serves (optional; default local+cloud)
    streams: bool  # optional; True if complete() accepts on_delta(text) and reports text as it is written

    async def complete(self, tier: str, operation: str, payload: dict) -> Reply: ...


def available_tiers(provider) -> tuple[str, ...]:
    models = getattr(provider, "models", None) or {"local": "local", "cloud": "cloud"}
    return tuple(t for t in TIER_ORDER if t in models)
