# Horizon Relay

An agent that uses local models for simple work, escalates complex reasoning to a large cloud model, and delegates easier subtasks back to local workers. Open WebUI provides the chat interface.

Read the [integration plan](INTEGRATION_PLAN.md) for the proposed Pipe integration, source map, relay contracts, implementation milestones, deployment prerequisites, and acceptance tests.

## Repository contents

- [`open-webui/`](open-webui/): unmodified Open WebUI v0.11.3 source imported from its official release archive.
- [`SOURCE.json`](SOURCE.json): upstream commit, archive checksum, and verification details. Upstream Git history is not included.
- [`INTEGRATION_PLAN.md`](INTEGRATION_PLAN.md): implementation roadmap.
- [`relay/`](relay/): **the relay**, a standalone Python package with its own [README](relay/README.md): Python API, CLI, routing policy, providers, and tests. It does not depend on Open WebUI.
- [`integrations/openwebui/`](integrations/openwebui/): the Open WebUI Pipe, a thin adapter over the package, plus its startup installer and tests.
- [`deployment/README.md`](deployment/README.md): CLI demo, container setup, and browser acceptance checklist.

## Status

Implemented: real local Horizon 0.9B workers and IFM 375B cloud planning, plus a separately labeled simulated provider. Both use graph validation, bounded repair/escalation, cancellation, per-run metrics, and Open WebUI status events. See [live model setup](deployment/LIVE_MODELS.md).

From the repository root, with Python 3.11+:

```sh
PYTHONPATH=relay/src python3 -m horizon_relay demo --scenario escalate --events
PYTHONPATH=relay/src python3 -m unittest discover -s relay/tests -v
PYTHONPATH=relay/src python3 -m unittest discover -s integrations/openwebui -v
```

The demo uses built-in bug reports and scripted responses; no real model calls or API keys are involved. Open WebUI's existing frontend is unchanged. Docker startup, the browser escalation demo, and saved chat reload have been verified locally. Browser Stop and multi-user behavior still need dedicated acceptance checks.

To use the relay outside Open WebUI (Python, `horizon-relay ask`, a different front end) or to change its routing rules, see [relay/README.md](relay/README.md).

For real conversations select **Horizon Relay**. Simple greetings and exactly validated extraction may stay local; other requests use cloud planning and synthesis. Generic worker answers are checked for output format, not proven factually correct. Real cloud calls use your configured API key; secrets remain outside Git and Docker images.

## Upstream notices

Open WebUI retains its [license](open-webui/LICENSE), [license history](open-webui/LICENSE_HISTORY), and [notices](open-webui/LICENSE_NOTICE). Those files apply to the vendored Open WebUI code.
