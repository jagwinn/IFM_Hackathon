"""
title: Horizon Relay (Simulated Demo)
author: Horizon Relay
version: 0.1.0
description: Bounded planner-worker demo using built-in bug reports. No real model calls.
"""

import asyncio
import os

from horizon_relay import Limits, RelayEngine, SimulatedProvider
from horizon_relay.providers import SCENARIOS, demo_sources
from horizon_relay.schemas import RelayError


class Pipe:
    def pipes(self):
        return [{"id": "demo", "name": "Horizon Relay (Simulated Demo)"}]

    async def pipe(self, body: dict, __event_emitter__=None, __task__=None,
                   __files__=None, __user__=None, __metadata__=None):
        if os.getenv("RELAY_PROVIDER", "simulated") != "simulated":
            raise ValueError("Only the simulated provider is implemented in this milestone")
        # UI title/tag requests must not create a plan or spend cloud calls.
        if __task__ is not None:
            async with asyncio.timeout(5):
                reply = await SimulatedProvider().complete("local", "background", {"task": str(__task__)})
            return reply.value
        if __files__ or body.get("files") or (__metadata__ or {}).get("files"):
            return "This prototype supports the built-in text demo only; remove attachments to continue."
        messages = body.get("messages", [])
        if not isinstance(messages, list) or any(not isinstance(m, dict) or not isinstance(m.get("content"), str) for m in messages):
            return "This prototype supports text messages only."
        latest = next((m["content"].strip() for m in reversed(messages) if m.get("role") == "user"), "")
        parts = latest.split()
        if not parts or parts[0] != "/relay-demo" or len(parts) > 2 or (len(parts) == 2 and parts[1] not in SCENARIOS):
            return ("**Horizon Relay — simulated prototype**\n\n"
                    "Run `/relay-demo`, `/relay-demo local`, `/relay-demo repair`, or `/relay-demo escalate`. "
                    "These use two built-in bug reports and scripted responses, not your conversation or a real model.")
        scenario = parts[1] if len(parts) == 2 else "standard"
        sources = demo_sources()
        if scenario == "local":
            sources = {"report_1": sources["report_1"]}
        enabled = os.getenv("CLOUD_ENABLED", "true").lower() == "true"
        try:
            result = await RelayEngine(SimulatedProvider(scenario, delay=0.15), Limits(cloud_enabled=enabled)).run(
                "Extract report 1 verbatim" if scenario == "local" else "Prioritize fixes for the supplied bug reports",
                sources, accept_local=scenario == "local", emit=__event_emitter__,
            )
            # Both streaming and non-streaming requests use Open WebUI's normal completion persistence.
            return (result.content + f'\n\nSimulated calls: {result.metrics["local_calls"]} local, '
                    f'{result.metrics["cloud_calls"]} cloud. Token usage: unavailable (simulation).')
        except RelayError as exc:
            return f"**Relay stopped — no complete answer.** {exc}"
