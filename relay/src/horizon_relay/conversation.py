"""Chat conversations -> relay requests. The latest user message is the goal; the whole conversation is the source."""

from dataclasses import dataclass

from .errors import RelayError

LOCAL_EXTRACT_COMMAND = "/local-extract"  # "/local-extract TEXT": copy TEXT exactly, checked, local only
MAX_CONVERSATION_CHARS = 24000


@dataclass(frozen=True)
class RelayRequest:
    goal: str
    sources: dict[str, str]
    local_extract: bool = False


def check_messages(messages) -> list[dict]:
    """OpenAI-style text messages: [{"role": "system" | "user" | "assistant", "content": str}, ...]."""
    if not isinstance(messages, list) or not messages or any(
        not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant")
        or not isinstance(m.get("content"), str) for m in messages
    ):
        raise RelayError("Horizon Relay currently accepts text-only conversations without tools or attachments")
    if len(transcript(messages)) > MAX_CONVERSATION_CHARS:
        raise RelayError(f"Conversation exceeds the prototype's {MAX_CONVERSATION_CHARS}-character limit; start a shorter chat")
    return messages


def transcript(messages) -> str:
    return "\n\n".join(f'{m["role"]}: {m["content"]}' for m in messages)


def to_request(messages) -> RelayRequest:
    check_messages(messages)
    goal = next((m["content"].strip() for m in reversed(messages) if m["role"] == "user"), "")
    if goal.startswith(LOCAL_EXTRACT_COMMAND + " "):
        return RelayRequest("Extract the text source verbatim", {"text": goal[len(LOCAL_EXTRACT_COMMAND) + 1:]}, True)
    return RelayRequest(goal, {"conversation": transcript(messages)})
