"""The entry point for using the relay from any application."""

import asyncio
from dataclasses import replace

from .conversation import check_messages, to_request
from .engine import RelayEngine, RunResult
from .errors import RelayError
from .events import EventCallback
from .policy import RoutingPolicy
from .providers import Provider
from .settings import Limits, RelaySettings


class Relay:
    """Answer requests with a small local model, escalating to a large cloud model only when needed.

        relay = Relay.from_env()
        result = await relay.chat([{"role": "user", "content": "Compare SQLite and Postgres"}])
        print(result.content, result.summary())
    """

    def __init__(self, provider: Provider, *, limits: Limits | None = None, policy: RoutingPolicy | None = None):
        self.provider = provider
        self.engine = RelayEngine(provider, limits, policy)

    @classmethod
    def from_settings(cls, settings: RelaySettings, *, policy: RoutingPolicy | None = None, transport=None) -> "Relay":
        try:
            from .providers.openai_compatible import OpenAICompatibleProvider
        except ModuleNotFoundError as exc:
            raise RelayError("Real models need httpx: pip install 'horizon-relay[live]'") from exc
        provider = OpenAICompatibleProvider(settings.local, settings.cloud, mid=settings.mid, transport=transport,
                                            timeout=settings.limits.call_timeout)
        return cls(provider, limits=settings.limits, policy=policy)

    @classmethod
    def from_env(cls, env=None, *, policy: RoutingPolicy | None = None) -> "Relay":
        """Real endpoints and routing settings from environment variables (RelaySettings.from_env and
        RoutingPolicy.from_env)."""
        return cls.from_settings(RelaySettings.from_env(env), policy=policy or RoutingPolicy.from_env(env))

    def with_policy(self, **overrides) -> "Relay":
        """The same models and limits with some routing settings changed, e.g. with_policy(cloud_mode="direct")."""
        return Relay(self.provider, limits=self.engine.limits, policy=replace(self.engine.policy, **overrides))

    async def chat(self, messages: list[dict], *, emit: EventCallback | None = None, on_delta=None) -> RunResult:
        """Answer the latest user message of an OpenAI-style conversation."""
        request = to_request(messages)
        return await self.run(request.goal, request.sources, local_extract=request.local_extract, emit=emit,
                              on_delta=on_delta)

    async def run(self, goal: str, sources: dict[str, str], *, local_extract: bool = False,
                  emit: EventCallback | None = None, on_delta=None) -> RunResult:
        """Answer `goal` using only the named text `sources`. `on_delta(call, text)` sees every chunk a model
        writes, for front ends that stream the text as it appears."""
        return await self.engine.run(goal, sources, local_extract=local_extract, emit=emit, on_delta=on_delta)

    async def complete(self, messages: list[dict], *, timeout: float = 120) -> str:
        """One direct call to the local model, without planning. For side requests such as chat titles."""
        check_messages(messages)
        async with asyncio.timeout(timeout):
            reply = await self.provider.complete("local", "direct", {"messages": messages})
        return reply.value
