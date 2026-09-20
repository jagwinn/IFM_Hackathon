"""Scripted demo runs over the built-in bug reports. No model or network calls."""

from .engine import RunResult
from .events import EventCallback
from .policy import RoutingPolicy
from .providers.simulated import SCENARIOS, SimulatedProvider, demo_sources
from .relay import Relay
from .settings import Limits

__all__ = ["SCENARIOS", "run_demo"]


async def run_demo(scenario: str = "standard", *, delay: float = 0, cloud_enabled: bool = True,
                   emit: EventCallback | None = None) -> RunResult:
    """standard: local answer not trusted; cloud plans, delegates, synthesizes. trusted: the local answer passes.
    local: exact extraction. repair: one local retry fixes a bad quote. escalate: two local failures, then only
    that subtask goes to the cloud."""
    sources = demo_sources()
    if scenario == "local":
        sources = {"report_1": sources["report_1"]}
    # The scripted scenarios exist to show planning and delegation, so they always plan.
    relay = Relay(SimulatedProvider(scenario, delay), limits=Limits(cloud_enabled=cloud_enabled),
                  policy=RoutingPolicy(cloud_mode="plan"))
    goal = {"local": "Extract report 1 verbatim", "trusted": "Summarize report 1 in one sentence"}.get(
        scenario, "Prioritize fixes for the supplied bug reports")
    return await relay.run(goal, sources, local_extract=scenario == "local", emit=emit)
