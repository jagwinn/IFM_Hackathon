"""Chat Completions client for any OpenAI-compatible server. Credentials never enter prompts or progress events."""

import json
import re
from urllib.parse import urlparse

import httpx

from .. import prompts
from ..errors import RelayError
from ..settings import Endpoint
from . import Reply


_devices: dict[str, dict] = {}  # server base URL -> its /device.json, fetched once per process


def describe_device(device: dict | None, show_index: bool) -> str:
    """"RTX 3070", "GPU 1 · RTX 3090", "GPU 0+1 · 2× RTX 3090" or "CPU" from a /device.json document."""
    if not isinstance(device, dict):
        return ""
    if device.get("kind") == "cpu":
        return "CPU"
    gpus = [g for g in device.get("gpus", []) if isinstance(g, dict) and isinstance(g.get("name"), str)]
    if not gpus:
        return ""
    names = [re.sub(r"^(NVIDIA\s+)?(GeForce\s+)?", "", g["name"]).strip() for g in gpus]
    label = names[0] if len(set(names)) == 1 else " + ".join(names)
    if len(names) > 1 and len(set(names)) == 1:
        label = f"{len(names)}× {label}"
    if show_index:
        label = "GPU " + "+".join(str(g.get("index")) for g in gpus) + " · " + label
    return label


class OpenAICompatibleProvider:
    simulated = False
    streams = True  # complete() accepts on_delta and reports text as the model writes it

    def __init__(self, local: Endpoint, cloud: Endpoint, *, mid: Endpoint | None = None, transport=None,
                 timeout: float = 180):
        self.local, self.cloud, self.mid, self.transport, self.timeout = local, cloud, mid, transport, timeout
        self.endpoints = {"local": local, **({"mid": mid} if mid else {}), "cloud": cloud}
        self.models = {tier: e.model for tier, e in self.endpoints.items()}

    async def locations(self) -> dict[str, str]:
        """Where each tier's model runs, for display: a configured location, the cloud host, or the hardware a
        llama.cpp server reports at /device.json (deployment/serve-model.sh). Never raises."""
        found = {}
        async with httpx.AsyncClient(timeout=2, transport=self.transport, follow_redirects=False) as client:
            for tier, endpoint in self.endpoints.items():
                if endpoint.location or tier == "cloud":
                    continue
                base = re.sub(r"/v1/?$", "", endpoint.url.rstrip("/"))
                if base not in _devices:
                    try:
                        response = await client.get(base + "/device.json")
                        if response.status_code == 200 and isinstance(response.json(), dict):
                            _devices[base] = response.json()
                    except Exception:
                        pass  # not a llama.cpp server started by serve-model.sh, or not reachable yet
                found[tier] = _devices.get(base)
        gpu_sets = [tuple(g.get("index") for g in d.get("gpus", []) if isinstance(g, dict))
                    for d in found.values() if isinstance(d, dict) and d.get("kind") == "gpu"]
        # Name the GPU when more than one is involved: a model spread over several, models on different
        # GPUs, or a model pinned to a GPU other than the first.
        show_index = len(set(gpu_sets)) > 1 or any(len(s) > 1 or any(i != 0 for i in s) for s in gpu_sets)
        return {tier: endpoint.location or (urlparse(endpoint.url).hostname or "" if tier == "cloud"
                                            else describe_device(found.get(tier), show_index))
                for tier, endpoint in self.endpoints.items()}

    async def complete(self, tier, operation, payload, on_delta=None):
        """on_delta(text, kind) is called with each chunk the model produces: kind is "thinking" while the
        model reasons and "output" once it writes the answer."""
        if tier not in self.endpoints:
            raise RelayError(f"No {tier} model is configured")
        endpoint = self.endpoints[tier]
        if tier == "cloud" and not endpoint.key:
            raise RelayError("Configure CLOUD_API_KEY before using cloud planning")
        structured = operation in ("attempt", "plan", "work", "solve", "critique")
        request = {"model": endpoint.model, "messages": prompts.messages(operation, payload),
                   "max_tokens": _max_tokens(tier, operation),
                   "temperature": payload.get("temperature", 0.2), "stream": bool(on_delta)}
        if on_delta:  # without this, streamed replies report no token counts
            request["stream_options"] = {"include_usage": True}
        if endpoint.reasoning_effort:
            request["chat_template_kwargs"] = {"reasoning_effort": endpoint.reasoning_effort}
        if tier != "cloud":
            if structured:
                request["response_format"] = {"type": "json_object"}
            if operation == "solve":  # token probabilities feed the token_uncertainty signal
                request["logprobs"], request["top_logprobs"] = True, 3
        headers = {"Content-Type": "application/json"}
        if endpoint.key:
            headers["Authorization"] = "Bearer " + endpoint.key
        url = endpoint.url.rstrip("/") + "/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport, follow_redirects=False) as client:
                if on_delta:
                    content, reasoning, logprobs, finish, usage = await _read_stream(client, url, request, headers, tier, on_delta)
                else:
                    response = await client.post(url, json=request, headers=headers)
                    if response.status_code != 200:
                        # Provider error bodies may echo a request/credential. Never relay them to chat.
                        raise RelayError(f"{tier.title()} model returned HTTP {response.status_code}; check endpoint configuration")
                    content, reasoning, logprobs, finish, usage = _read_response(response, tier)
        except httpx.TimeoutException as exc:
            raise RelayError(f"{tier.title()} model timed out") from exc
        except httpx.HTTPError as exc:
            raise RelayError(f"Cannot connect to the {tier} model endpoint") from exc
        if not isinstance(content, str) or not content.strip() or finish == "length":
            raise RelayError(f"{tier.title()} response was empty or reached its token limit")
        thinking, content = _split_thinking(content, reasoning)
        value = content
        if structured:
            clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip()
            try:
                value = json.loads(clean)
            except ValueError:
                # Let the engine's bounded plan/worker validation handle malformed JSON.
                value = content
        def count(name):
            value = usage.get(name)
            return value if type(value) is int and value >= 0 else None
        return Reply(value, endpoint.model, count("prompt_tokens"), count("completion_tokens"), logprobs,
                     thinking=thinking, text=content)


def _max_tokens(tier: str, operation: str) -> int:
    """Output budget per call. Models that reason before answering need room for the reasoning too:
    the 4B regularly ran past 2048 tokens on harder requests, and the cloud model needs far more."""
    if tier == "cloud":
        return 8192 if operation == "work" else 16384
    return 4096 if tier == "mid" else 2048


def _read_response(response, tier) -> tuple[str, str, list | None, str | None, dict]:
    """A complete (non-streaming) reply: (content, reasoning, token logprobs, finish reason, usage)."""
    try:
        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]
        usage = data.get("usage") or {}
        return (message.get("content"), message.get("reasoning_content") or "", _token_logprobs(choice),
                choice.get("finish_reason"), usage if isinstance(usage, dict) else {})
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RelayError(f"{tier.title()} endpoint returned an invalid completion") from exc


async def _read_stream(client, url, request, headers, tier, on_delta) -> tuple[str, str, list | None, str | None, dict]:
    """Server-sent chunks, passed to on_delta as they arrive and assembled into the same five parts."""
    content, reasoning, tokens, finish, usage = "", "", [], None, {}
    # K2 writes its reasoning inline and ends it with a tokenizer marker; before that marker the text is thinking.
    marker, thinking = re.compile(r"</ifm\|think(?:_[a-z]+)?>"), True
    async with client.stream("POST", url, json=request, headers=headers) as response:
        if response.status_code != 200:
            await response.aread()
            raise RelayError(f"{tier.title()} model returned HTTP {response.status_code}; check endpoint configuration")
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                event = json.loads(chunk)
            except ValueError:
                continue  # keep-alive or a partial line
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                finish = choice.get("finish_reason") or finish
                tokens.extend((choice.get("logprobs") or {}).get("content") or [])
                reasoning_delta = delta.get("reasoning_content")
                if isinstance(reasoning_delta, str) and reasoning_delta:
                    reasoning += reasoning_delta
                    on_delta(reasoning_delta, "thinking")
                text = delta.get("content")
                if isinstance(text, str) and text:
                    content += text
                    if not thinking:
                        on_delta(text, "output")
                        continue
                    match = marker.search(content)
                    if not match:
                        on_delta(text, "thinking")
                    else:  # the marker may arrive mid-chunk: split this chunk, dropping the marker itself
                        thinking = False
                        tail = content[match.end():]
                        head = text[:max(0, len(text) - len(tail) - len(match.group(0)))]
                        if head.strip():
                            on_delta(head, "thinking")
                        if tail:
                            on_delta(tail, "output")
    return content, reasoning, _token_logprobs({"logprobs": {"content": tokens}}), finish, usage


def _split_thinking(content: str, reasoning: str = "") -> tuple[str, str]:
    """(what the model reasoned, the answer it settled on). K2 writes its reasoning inline, ending at a
    tokenizer marker, instead of filling a separate reasoning field."""
    thinking = reasoning
    match = re.match(r"\s*<think>(.*?)</think>\s*", content, flags=re.S)
    if match:
        thinking, content = (thinking + "\n" + match.group(1)).strip(), content[match.end():]
    parts = re.split(r"</ifm\|think(?:_[a-z]+)?>", content, maxsplit=1)
    if len(parts) == 2:
        thinking, content = (thinking + "\n" + parts[0]).strip(), parts[1]
    return thinking.strip(), content.strip()


def _token_logprobs(choice) -> list | None:
    """OpenAI-style choice.logprobs.content as [{"t", "lp", "top"}], or None if absent or malformed."""
    try:
        content = (choice.get("logprobs") or {}).get("content")
        if not content:
            return None
        return [{"t": str(t["token"]), "lp": float(t["logprob"]),
                 "top": [[str(a["token"]), float(a["logprob"])] for a in t.get("top_logprobs") or []]}
                for t in content]
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
