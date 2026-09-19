# Real Horizon models

The live Pipe uses the supplied conversation, not the scripted demo. Select **Horizon Relay** in the model picker. The separately labeled **Horizon Relay (Simulated Demo)** remains available for deterministic tests.

## Models and runtimes

- Cloud: `IFM/K2-Horizon-375B-A23B` at `https://api.ifm.ai/v1`.
- Local: IFM's [K2-Horizon-0.9B-GGUF](https://huggingface.co/IFM/K2-Horizon-0.9B-GGUF), revision `c9e2c6d99a9c682cdc2c4f439c480c07a6ded35c`, file `K2-Horizon-1B-BF16.gguf`. The filename says 1B; this is the official 0.9B-class repository.
- Local runtime: IFM's [compatible llama.cpp fork](https://github.com/MBZUAI-IFM/llama.cpp/tree/model/K2Horizon), pinned to `42adf019f76013dac873b5b43950d54d5ab27216` and built in `Dockerfile.local-model`.

Docker Model Runner downloaded the model successfully on the development Mac, but its bundled llama.cpp b9879 rejected the `k2-horizon` architecture. The live Compose setup therefore runs the compatible fork in a separate Linux ARM64 CPU container. Native Metal acceleration is not used by this container.

The active local weights are **Q4_K_M**, converted locally from the official BF16 file using the compatible fork's `llama-quantize`. Quantization reduced tensor size from 2056.83 MiB to 632.77 MiB. BF16 was slow in this CPU setup; Q4_K_M retains the same model with lower weight precision. Point `LOCAL_MODEL_PATH` at the quantized output, mounted read-only by Compose.

## Start

1. Download the official GGUF file from the repository above. Alternatively, Docker Model Runner can download it with `docker model pull hf.co/IFM/K2-Horizon-0.9B-GGUF:BF16`; inspect its local bundle path for `LOCAL_MODEL_PATH`.
2. In the private, ignored `.env`, configure `CLOUD_API_KEY`, `LOCAL_MODEL_PATH`, and `RELAY_PROVIDER=live`. Retain your existing `WEBUI_SECRET_KEY` and auth mode. Never commit the populated `.env`. The inference image also includes `llama-quantize`; it can convert the source GGUF with `llama-quantize input.gguf output.gguf Q4_K_M` before mounting it for inference.
3. Build and launch from the repository root:

   ```sh
   docker compose --env-file .env -f deployment/compose.relay.yaml -f deployment/compose.live.yaml -p horizon-relay up --build -d
   ```

4. Update the installed `horizon_relay` Function with the current `integrations/openwebui/horizon_relay_pipe.py`, then select **Horizon Relay** in a new chat at <http://localhost:3000>.

The key is passed only to the backend cloud adapter at runtime. It is neither in the Docker image nor in model prompts, progress events, or the repository. The local inference service has no published host port; the UI remains bound to localhost.

## Try it

- `Hello` — a small greeting may complete locally.
- `/local-extract The release date is Friday.` — local exact extraction, checked against the source.
- `Plan a two-day prototype for a campus lost-and-found app. Compare two storage choices, suggest a minimal feature set, and identify the biggest implementation risks.` — local routing, cloud planning, local subtasks, cloud synthesis.

General tasks use a conservative escalation policy. A small model's confidence is not enough to bypass planning: only greetings and exactly validated extraction currently qualify for the local-only path. Local workers produce structured answers checked for shape; this does **not** prove factual correctness. The large model reviews their results during synthesis.

Calls have a 120-second timeout and the overall run has a 480-second deadline. The existing 6-task, 5-cloud-call, 2-cloud-worker-call limits still apply. JSON errors receive the engine's bounded repair attempts; transport failures stop with a sanitized error. No hidden HTTP retries spend additional cloud calls. Requests do not stream model tokens yet; progress arrives live and final text arrives after synthesis.

Only text conversations are supported, up to the configured character limits. There is no browsing, code execution, arbitrary tool use, or attachment parsing. Background title/tag requests use the local model and are separate from the visible turn's metrics. Token counts come from provider usage fields; missing counts are shown as unavailable.

## Tests

```sh
python3 -m pip install -e './relay[live]'
python3 -m unittest discover -s relay/tests -v
```

HTTP adapter tests use mock transports and incur no API costs. Real smoke checks are performed separately against the configured endpoints.

Verified on September 19, 2026: local exact extraction completed in 1.5 seconds with one local call and zero cloud calls. A browser request comparing databases completed with four local calls (routing plus three subtasks), two cloud calls (planning and synthesis), and 3,954 reported tokens. Its saved history retained all progress events. These are smoke-test observations, not a cost or quality benchmark against an all-cloud baseline.
