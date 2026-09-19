# Horizon Relay

An agent that uses local models for simple work, escalates complex reasoning to a large cloud model, and delegates easier subtasks back to local workers. Open WebUI provides the chat interface.

Read the [integration plan](INTEGRATION_PLAN.md) for the proposed Pipe integration, source map, relay contracts, implementation milestones, deployment prerequisites, and acceptance tests.

## Repository contents

- [`open-webui/`](open-webui/): unmodified Open WebUI v0.11.3 source imported from its official release archive.
- [`SOURCE.json`](SOURCE.json): upstream commit, archive checksum, and verification details. Upstream Git history is not included.
- [`INTEGRATION_PLAN.md`](INTEGRATION_PLAN.md): implementation roadmap.
- [`relay/`](relay/): dependency-free Python engine, simulated provider, and tests.
- [`integrations/openwebui/horizon_relay_pipe.py`](integrations/openwebui/horizon_relay_pipe.py): Open WebUI Pipe adapter.
- [`deployment/README.md`](deployment/README.md): CLI demo, container setup, and browser acceptance checklist.

## Status

Implemented: a runnable **simulated** local → cloud planner → local workers → synthesis lifecycle, graph and evidence validation, bounded repair/escalation, cancellation, per-run metrics, and Open WebUI status events.

From the repository root, with Python 3.11+:

```sh
PYTHONPATH=relay/src python3 -m horizon_relay --scenario escalate --events
PYTHONPATH=relay/src python3 -m unittest discover -s relay/tests -v
```

The demo uses built-in bug reports and scripted responses; no real model calls or API keys are involved. Open WebUI's existing frontend is unchanged. Docker startup, the browser escalation demo, and saved chat reload have been verified locally. Browser Stop and multi-user behavior still need dedicated acceptance checks.

Next: run the [Open WebUI setup](deployment/README.md), then implement the real local/cloud provider adapters after confirming the endpoints and model IDs.

## Upstream notices

Open WebUI retains its [license](open-webui/LICENSE), [license history](open-webui/LICENSE_HISTORY), and [notices](open-webui/LICENSE_NOTICE). Those files apply to the vendored Open WebUI code.
