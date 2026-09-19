from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Limits:
    max_tasks: int = 6
    local_concurrency: int = 1
    cloud_calls: int = 5
    cloud_worker_calls: int = 2
    cloud_enabled: bool = True
    call_timeout: float = 30.0
    run_timeout: float = 180.0

    def __post_init__(self):
        for name in ("max_tasks", "local_concurrency", "cloud_calls", "cloud_worker_calls"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name in ("max_tasks", "local_concurrency") else 0):
                raise ValueError(f"Invalid limit: {name}")
        if self.max_tasks > 6 or self.local_concurrency > 2:
            raise ValueError("Prototype supports at most 6 tasks and 2 local workers")
        for name in ("call_timeout", "run_timeout"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid timeout: {name}")
