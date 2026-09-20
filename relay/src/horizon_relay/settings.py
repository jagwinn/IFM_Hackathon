"""Limits and endpoint configuration. RelaySettings.from_env() is the only place environment variables are read."""

from dataclasses import dataclass, field
import math
import os
from typing import Mapping


@dataclass(frozen=True)
class Limits:
    max_tasks: int = 8  # a large goal can be split into more pieces than a small one
    local_concurrency: int = 2  # local subtasks running at once, alongside whatever the cloud is doing
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
        if self.max_tasks > 12 or self.local_concurrency > 4:
            raise ValueError("Prototype supports at most 12 tasks and 4 local workers")
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
    reasoning_effort: str = ""  # "low", "medium", "high" for models that take it; higher means more thinking to show


@dataclass(frozen=True)
class RelaySettings:
    local: Endpoint  # the smallest model: triage and easy subtasks
    cloud: Endpoint  # the large model: planning, synthesis, last-resort subtasks
    # Local models writing code or long drafts are slow, and a delegated plan runs several of them.
    limits: Limits = Limits(call_timeout=180, run_timeout=900)
    mid: Endpoint | None = None  # optional middle tier between local and cloud

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "RelaySettings":
        """LOCAL_BASE_URL, LOCAL_MODEL, LOCAL_API_KEY, CLOUD_BASE_URL, CLOUD_MODEL, CLOUD_API_KEY, CLOUD_ENABLED,
        optionally MID_BASE_URL, MID_MODEL, MID_API_KEY for a middle tier, and LOCAL_LOCATION, MID_LOCATION,
        CLOUD_LOCATION to name where each model runs instead of detecting it, RELAY_REASONING_EFFORT
        (per model: LOCAL_/MID_/CLOUD_REASONING_EFFORT) for how much the models think before answering, and
        RELAY_CALL_TIMEOUT / RELAY_RUN_TIMEOUT in seconds."""
        env = os.environ if env is None else env
        effort = env.get("RELAY_REASONING_EFFORT", "low")
        mid = None
        if env.get("MID_BASE_URL"):
            mid = Endpoint(env["MID_BASE_URL"], env.get("MID_MODEL", "IFM/K2-Horizon-4B"), env.get("MID_API_KEY", ""),
                           env.get("MID_LOCATION", ""), env.get("MID_REASONING_EFFORT", effort))
        return cls(
            Endpoint(env.get("LOCAL_BASE_URL", "http://model-runner.docker.internal/engines/v1"),
                     env.get("LOCAL_MODEL", "hf.co/IFM/K2-Horizon-0.9B-GGUF:BF16"),
                     env.get("LOCAL_API_KEY", ""), env.get("LOCAL_LOCATION", ""),
                     env.get("LOCAL_REASONING_EFFORT", effort)),
            Endpoint(env.get("CLOUD_BASE_URL", "https://api.ifm.ai/v1"),
                     env.get("CLOUD_MODEL", "IFM/K2-Horizon-375B-A23B"),
                     env.get("CLOUD_API_KEY", ""), env.get("CLOUD_LOCATION", ""),
                     env.get("CLOUD_REASONING_EFFORT", "")),
            Limits(call_timeout=float(env.get("RELAY_CALL_TIMEOUT", 180)),
                   run_timeout=float(env.get("RELAY_RUN_TIMEOUT", 900)),
                   cloud_enabled=env.get("CLOUD_ENABLED", "true").lower() == "true"),
            mid,
        )
