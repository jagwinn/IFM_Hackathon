"""Limits and endpoint configuration. RelaySettings.from_env() is the only place environment variables are read."""

from dataclasses import dataclass, field
import math
import os
from typing import Mapping


@dataclass(frozen=True)
class Limits:
    max_tasks: int = 6
    local_concurrency: int = 1
    cloud_calls: int = 5  # includes planning and the final synthesis
    cloud_worker_calls: int = 2  # subtasks that escalated to the cloud
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


@dataclass(frozen=True)
class Endpoint:
    """An OpenAI-compatible Chat Completions server."""
    url: str
    model: str
    key: str = field(default="", repr=False)
    location: str = ""  # shown next to the model, e.g. "IFM API"; empty = detect (see OpenAICompatibleProvider.locations)


@dataclass(frozen=True)
class RelaySettings:
    local: Endpoint  # the smallest model: triage and easy subtasks
    cloud: Endpoint  # the large model: planning, synthesis, last-resort subtasks
    limits: Limits = Limits(call_timeout=120, run_timeout=480)
    mid: Endpoint | None = None  # optional middle tier between local and cloud

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "RelaySettings":
        """LOCAL_BASE_URL, LOCAL_MODEL, LOCAL_API_KEY, CLOUD_BASE_URL, CLOUD_MODEL, CLOUD_API_KEY, CLOUD_ENABLED,
        optionally MID_BASE_URL, MID_MODEL, MID_API_KEY for a middle tier, and LOCAL_LOCATION, MID_LOCATION,
        CLOUD_LOCATION to name where each model runs instead of detecting it."""
        env = os.environ if env is None else env
        mid = None
        if env.get("MID_BASE_URL"):
            mid = Endpoint(env["MID_BASE_URL"], env.get("MID_MODEL", "IFM/K2-Horizon-4B"), env.get("MID_API_KEY", ""),
                           env.get("MID_LOCATION", ""))
        return cls(
            Endpoint(env.get("LOCAL_BASE_URL", "http://model-runner.docker.internal/engines/v1"),
                     env.get("LOCAL_MODEL", "hf.co/IFM/K2-Horizon-0.9B-GGUF:BF16"),
                     env.get("LOCAL_API_KEY", ""), env.get("LOCAL_LOCATION", "")),
            Endpoint(env.get("CLOUD_BASE_URL", "https://api.ifm.ai/v1"),
                     env.get("CLOUD_MODEL", "IFM/K2-Horizon-375B-A23B"),
                     env.get("CLOUD_API_KEY", ""), env.get("CLOUD_LOCATION", "")),
            Limits(call_timeout=120, run_timeout=480,
                   cloud_enabled=env.get("CLOUD_ENABLED", "true").lower() == "true"),
            mid,
        )
