# Real Horizon models

The live Pipe uses the supplied conversation, not the scripted demo. Select **Horizon Relay** in the model picker. The separately labeled **Horizon Relay (Simulated Demo)** remains available for deterministic tests.

## Models and runtimes

- Cloud: `IFM/K2-Horizon-375B-A23B` at `https://api.ifm.ai/v1`.
- Local: IFM's [K2-Horizon-0.9B-GGUF](https://huggingface.co/IFM/K2-Horizon-0.9B-GGUF), revision `c9e2c6d99a9c682cdc2c4f439c480c07a6ded35c`, file `K2-Horizon-1B-BF16.gguf`. The filename says 1B; this is the official 0.9B-class repository.
- Local 4B: IFM's [K2-Horizon-3.7B-GGUF](https://huggingface.co/IFM/K2-Horizon-3.7B-GGUF), revision `67b824b8079635b77e19a2a4cd2682a9716478c1`, file `K2-Horizon-4B-BF16.gguf`, quantized to Q4_K_M (9648 MiB to 2999 MiB) and served as `IFM/K2-Horizon-4B` by the `local-model-4b` service.
- Local runtime: IFM's [compatible llama.cpp fork](https://github.com/MBZUAI-IFM/llama.cpp/tree/model/K2Horizon), pinned to `42adf019f76013dac873b5b43950d54d5ab27216` and built in `Dockerfile.local-model`.

Docker Model Runner downloaded the model successfully on the development Mac, but its bundled llama.cpp b9879 rejected the `k2-horizon` architecture. The live Compose setup therefore runs the compatible fork in a separate Linux ARM64 CPU container. Native Metal acceleration is not used by this container.

The active local weights are **Q4_K_M**, converted locally from the official BF16 file using the compatible fork's `llama-quantize`. Quantization reduced tensor size from 2056.83 MiB to 632.77 MiB. BF16 was slow in this CPU setup; Q4_K_M retains the same model with lower weight precision. Point `LOCAL_MODEL_PATH` at the quantized output, mounted read-only by Compose.

## Decision graph

Each **Horizon Relay** reply carries an interactive graph between the status line and the answer: which model handled each step and why, every failed check, and every delegation down or escalation up. It updates live during the run and is saved with the chat. Collapse or hide it from its header. The relay uses the 4B as a middle tier (`MID_BASE_URL` in `compose.live.yaml`).

## Direct model connections

Besides the Pipe, Open WebUI lists three OpenAI-compatible connections from `compose.live.yaml`: the local 0.9B, the local 4B, and the IFM cloud API (authenticated with `CLOUD_API_KEY`). Open WebUI stores connection settings in its database on first start; after that, change them in **Admin Settings > Connections** or delete the `openai.*` rows from its `config` table to reapply the environment.

## GPU

Add `-f deployment/compose.gpu.yaml` to run both local models on an NVIDIA GPU. `CUDA_ARCHS` (default `86`, RTX 30-series) selects the compiled architecture. With the BF16 0.9B and Q4_K_M 4B loaded, an RTX 3070 uses about 6.7 of 8 GB. The CUDA image needs `--gpus all` for one-off commands such as `llama-quantize`. Each model is pinned to one GPU: set `LOCAL_MODEL_GPU` and `MID_MODEL_GPU` in `.env` to host indexes from `nvidia-smi -L` (both default to 0). Each model container reports its hardware at `/device.json` (`deployment/serve-model.sh`), and the decision graph labels its lane with it, adding the GPU number when more than one GPU is in use. Set `LOCAL_LOCATION`, `MID_LOCATION` or `CLOUD_LOCATION` to override a label.

## Start

1. Download the official GGUF file from the repository above. Alternatively, Docker Model Runner can download it with `docker model pull hf.co/IFM/K2-Horizon-0.9B-GGUF:BF16`; inspect its local bundle path for `LOCAL_MODEL_PATH`.
2. In the private, ignored `.env`, configure `CLOUD_API_KEY`, `LOCAL_MODEL_PATH`, `LOCAL_MODEL_4B_PATH`, and `RELAY_PROVIDER=live`. Retain your existing `WEBUI_SECRET_KEY` and auth mode. Never commit the populated `.env`. The inference image also includes `llama-quantize`; it can convert the source GGUF with `llama-quantize input.gguf output.gguf Q4_K_M` before mounting it for inference.
3. Build and launch from the repository root:

   ```sh
   docker compose --env-file .env -f deployment/compose.relay.yaml -f deployment/compose.live.yaml -p horizon-relay up --build -d
   ```

4. Select **Horizon Relay** in a new chat at <http://localhost:3000>. The container installs or updates the `horizon_relay` Function from `integrations/openwebui/horizon_relay_pipe.py` on every start (`integrations/openwebui/sync_pipe.py`), so edits in the admin Functions editor are replaced by the image's version on restart.

The key is passed only to the backend cloud adapter at runtime. It is neither in the Docker image nor in model prompts, progress events, or the repository. The local inference services have no published host port. The UI binds to localhost unless `WEBUI_BIND` is set (for example `0.0.0.0` for LAN access); keep `WEBUI_AUTH=true` when exposing it, because the UI holds the cloud key.

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
python3 -m unittest discover -s integrations/openwebui -v
```

HTTP adapter tests use mock transports and incur no API costs. Real smoke checks are performed separately against the configured endpoints.

Verified on September 19, 2026: local exact extraction completed in 1.5 seconds with one local call and zero cloud calls. A browser request comparing databases completed with four local calls (routing plus three subtasks), two cloud calls (planning and synthesis), and 3,954 reported tokens. Its saved history retained all progress events. These are smoke-test observations, not a cost or quality benchmark against an all-cloud baseline.
