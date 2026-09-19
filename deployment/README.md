# Run the simulated prototype

This milestone makes **no model API calls**. It exercises the real scheduling, checks, budgets, and failure handling with a deterministic provider and two built-in bug reports. It does not answer arbitrary questions yet.

## Run without Open WebUI

From the repository root, with Python 3.11 or newer:

```sh
PYTHONPATH=relay/src python3 -m horizon_relay --scenario standard --events
PYTHONPATH=relay/src python3 -m horizon_relay --scenario local --no-cloud
PYTHONPATH=relay/src python3 -m horizon_relay --scenario repair --events
PYTHONPATH=relay/src python3 -m horizon_relay --scenario escalate --events
PYTHONPATH=relay/src python3 -m unittest discover -s relay/tests -v
```

No installation or third-party Python libraries are needed for those commands. Status events go to stderr as JSONL; the answer and measured orchestration timings go to stdout. Token counts remain unknown because no model generated tokens.

Alternatively install with `python3 -m pip install -e ./relay`, then run `horizon-relay --events`.

## Run inside Open WebUI

Requires a working Docker installation. The container supplies Open WebUI's supported Python runtime and UI build.

1. Copy `.env.example` to `.env`. Generate a secret with `python3 -c 'import secrets; print(secrets.token_hex(32))'` and put it in `WEBUI_SECRET_KEY` in `.env`.
2. From the repository root run:

   ```sh
   docker compose --env-file .env -f deployment/compose.relay.yaml up --build -d
   ```

3. Open <http://localhost:3000> and create the initial administrator account.
4. In the admin **Functions** editor, create a Function with ID `horizon_relay` and paste the contents of `integrations/openwebui/horizon_relay_pipe.py`. Save and enable it. The derived image already installs the Python package the Pipe imports.
5. Select **Horizon Relay (Simulated Demo)** in a new chat. Send one of the commands below as the entire message.

| Message | Expected behavior |
| --- | --- |
| `/relay-demo` | Local attempt → cloud plan → two local extractions → local grouping → cloud synthesis. |
| `/relay-demo local` | One validated extraction, no simulated cloud calls. |
| `/relay-demo repair` | First extraction fails once; a local retry fixes it. |
| `/relay-demo escalate` | First extraction fails twice locally; only that subtask escalates. |

Progress appears in the existing expandable status history. All events and answers are labeled simulated. The final answer uses the ordinary completion return path, so both streaming and non-streaming requests receive the answer; this milestone returns one final text chunk rather than token-by-token model streaming.

Set unrelated tools and global filters aside for this demo. The Pipe handles title/tag/follow-up requests locally with scripted output; once a real local endpoint exists, configure it as Open WebUI's task model. `CLOUD_ENABLED=false` disables even simulated cloud stages while permitting the local scenario.

## Browser acceptance checklist

These require a running Open WebUI instance and are not covered by direct Pipe unit tests:

- Model appears after the Function is enabled.
- A standard run shows progress followed by one final answer.
- Saved chat reload retains the answer and status history; temporary chat behavior remains unchanged.
- Stop during the demo prevents further work and leaves the chat usable.
- Regenerate starts a new run; simultaneous users do not share run state.

## What is still deferred

Real HTTP provider adapters, model downloads, arbitrary text tasks, uploads/RAG, model token streaming, custom task-graph UI, and all-cloud benchmarking. Real model IDs and the cloud API URL are intentionally not guessed.

The Docker image build and browser acceptance checklist have not been run on the development machine because Docker is unavailable there. The engine and Pipe are tested directly with Python.
