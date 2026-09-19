"""Normalize real chat requests without replacing them with demo fixtures."""

from .config import Limits
from .engine import RelayEngine
from .schemas import RelayError


async def live_chat(body, emit=None, task=None):
    from .live import LiveProvider
    import os
    provider = LiveProvider.from_env()
    messages = body.get("messages", [])
    if not isinstance(messages, list) or not messages or any(
        not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant")
        or not isinstance(m.get("content"), str) for m in messages
    ):
        raise RelayError("Horizon Relay currently accepts text-only conversations without tools or attachments")
    conversation = "\n\n".join(f'{m["role"]}: {m["content"]}' for m in messages)
    if len(conversation) > 24000:
        raise RelayError("Conversation exceeds the prototype's 24000-character limit; start a shorter chat")
    if task is not None:
        import asyncio
        async with asyncio.timeout(120):
            reply = await provider.complete("local", "background", {"messages": messages})
        return reply.value
    goal = next((m["content"].strip() for m in reversed(messages) if m["role"] == "user"), "")
    local_extract = goal.startswith("/local-extract ")
    sources = {"conversation": conversation}
    if local_extract:
        sources = {"text": goal[len("/local-extract "):]}
        goal = "Extract the text source verbatim"
    result = await RelayEngine(provider, Limits(call_timeout=120, run_timeout=480,
        cloud_enabled=os.getenv("CLOUD_ENABLED", "true").lower() == "true")).run(
            goal, sources, accept_local=local_extract, emit=emit)
    calls = result.metrics["calls"]
    measured = all(c["prompt_tokens"] is not None and c["completion_tokens"] is not None for c in calls)
    usage = str(sum(c["prompt_tokens"] + c["completion_tokens"] for c in calls)) if measured else "unavailable for some calls"
    return (result.content + f'\n\n---\nRelay: {result.metrics["local_calls"]} local calls · '
            f'{result.metrics["cloud_calls"]} cloud calls · tokens: {usage}.')
