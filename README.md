# Horizon Relay

An agent that uses local models for simple work, escalates complex reasoning to a large cloud model, and delegates easier subtasks back to local workers. Open WebUI provides the chat interface.

Read the [integration plan](INTEGRATION_PLAN.md) for the proposed Pipe integration, source map, relay contracts, implementation milestones, deployment prerequisites, and acceptance tests.

## Repository contents

- [`open-webui/`](open-webui/): unmodified Open WebUI v0.11.3 source imported from its official release archive.
- [`SOURCE.json`](SOURCE.json): upstream commit, archive checksum, and verification details. Upstream Git history is not included.
- [`INTEGRATION_PLAN.md`](INTEGRATION_PLAN.md): implementation roadmap.

## Status

Completed: source download and inspection, provenance record, integration plan.

Next: implement a Horizon Relay Pipe and independent Python engine with a clearly labeled fake provider, then connect the local model and cloud planner.

The application has not been installed or started, and no Horizon Relay implementation has been added yet.

## Upstream notices

Open WebUI retains its [license](open-webui/LICENSE), [license history](open-webui/LICENSE_HISTORY), and [notices](open-webui/LICENSE_NOTICE). Those files apply to the vendored Open WebUI code.
