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

    def __init__(self, local: Endpoint, cloud: Endpoint, *, mid: Endpoint | None = None, transport=None):
        self.local, self.cloud, self.mid, self.transport = local, cloud, mid, transport
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

    async def complete(self, tier, operation, payload):
        if tier not in self.endpoints:
            raise RelayError(f"No {tier} model is configured")
        endpoint = self.endpoints[tier]
        if tier == "cloud" and not endpoint.key:
            raise RelayError("Configure CLOUD_API_KEY before using cloud planning")
        structured = operation in ("attempt", "plan", "work", "solve", "critique")
        request = {"model": endpoint.model, "messages": prompts.messages(operation, payload),
                   "max_tokens": _max_tokens(tier, operation),
                   "temperature": payload.get("temperature", 0.2), "stream": False}
        if tier != "cloud":
            request["chat_template_kwargs"] = {"reasoning_effort": "low"}
            if structured:
                request["response_format"] = {"type": "json_object"}
            if operation == "solve":  # token probabilities feed the token_uncertainty signal
                request["logprobs"], request["top_logprobs"] = True, 3
        headers = {"Content-Type": "application/json"}
        if endpoint.key:
            headers["Authorization"] = "Bearer " + endpoint.key
        try:
            async with httpx.AsyncClient(timeout=120, transport=self.transport, follow_redirects=False) as client:
                response = await client.post(endpoint.url.rstrip("/") + "/chat/completions", json=request, headers=headers)
        except httpx.TimeoutException as exc:
            raise RelayError(f"{tier.title()} model timed out") from exc
        except httpx.HTTPError as exc:
            raise RelayError(f"Cannot connect to the {tier} model endpoint") from exc
        if response.status_code != 200:
            # Provider error bodies may echo a request/credential. Never relay them to chat.
            raise RelayError(f"{tier.title()} model returned HTTP {response.status_code}; check endpoint configuration")
        try:
            data = response.json()
            choice = data["choices"][0]
            content = choice["message"].get("content")
            if not isinstance(content, str) or not content.strip() or choice.get("finish_reason") == "length":
                raise RelayError(f"{tier.title()} response was empty or reached its token limit")
            usage = data.get("usage") or {}
            if not isinstance(usage, dict):
                usage = {}
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RelayError(f"{tier.title()} endpoint returned an invalid completion") from exc
        content = re.sub(r"^\s*<think>.*?</think>\s*", "", content, flags=re.S).strip()
        # IFM's llama.cpp fork currently returns K2 reasoning inline, ending at
        # this tokenizer marker instead of filling a separate reasoning field.
        content = re.split(r"</ifm\|think(?:_[a-z]+)?>", content, maxsplit=1)[-1].strip()
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
        return Reply(value, endpoint.model, count("prompt_tokens"), count("completion_tokens"), _token_logprobs(choice))


def _max_tokens(tier: str, operation: str) -> int:
    """Output budget per call. The cloud model reasons before answering, so it needs far more room."""
    if tier == "cloud":
        return 8192 if operation == "work" else 16384
    return 2048


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
