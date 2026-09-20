"""Progress events. The relay emits these; each front end decides how to display them."""

from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class RelayEvent:
    run_id: str
    seq: int  # 1, 2, 3... within a run; callbacks receive events one at a time, in order
    name: str  # machine-readable step, e.g. "plan_created", "task_invalid", "run_completed"
    message: str  # human-readable description of the step
    done: bool = False  # True on the final event of a run (completed, failed or cancelled)
    simulated: bool = False  # True when the provider is scripted rather than a real model
    data: dict[str, Any] = field(default_factory=dict)  # step details: tier, task_id, attempt, error, tasks, metrics

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


EventCallback = Callable[[RelayEvent], Awaitable[None]]
