# Horizon Relay

A standalone Python package that answers requests with a small **local** model and brings in a large **cloud** model only when needed. It has no dependency on Open WebUI; the Open WebUI Pipe in [`integrations/openwebui/`](../integrations/openwebui/) is one front end among many possible ones.

## How a request flows

```
request
   │
   ▼
1. Local answers ── each local model, smallest first (0.9B, then 4B), answers twice and rates itself;
   │                a critic (the next larger local model) checks it; six signals give a risk score.
   │                Rules or score ≥ threshold → next model up.        Trusted → answer (0 cloud calls)
   ▼
2. Cloud ────────── cloud_mode="plan": split into subtasks, delegate them back to local models,
                    check every result, escalate only failed subtasks, synthesize the answer.
                    cloud_mode="direct": the cloud answers in one call.
```

Limits bound every run: tasks, cloud calls (one is always kept for the final answer), cloud subtask calls, per-call and per-run timeouts. Cancelling the caller's task stops all work.

## Where to change things

| To change | Edit |
| --- | --- |
| When the local model may answer alone, retries, escalation | [`policy.py`](src/horizon_relay/policy.py) (`RoutingPolicy` and its rules) |
| What the models are told at each step | [`prompts.py`](src/horizon_relay/prompts.py) |
| Kinds of subtask results and how they are checked | [`results.py`](src/horizon_relay/results.py) (`RESULT_TYPES`) |
| Call, task and time limits | [`settings.py`](src/horizon_relay/settings.py) (`Limits`) |
| Endpoints, models, keys (environment variables) | [`settings.py`](src/horizon_relay/settings.py) (`RelaySettings.from_env`) |
| The decision graph (data / drawing) | [`graph.py`](src/horizon_relay/graph.py) / [`ui/graph.html`](src/horizon_relay/ui/graph.html) |
| How a chat conversation becomes a request (`/local-extract`) | [`conversation.py`](src/horizon_relay/conversation.py) |
| Talking to a different model API | [`providers/`](src/horizon_relay/providers/) |

The rest is machinery you rarely need to touch: [`engine.py`](src/horizon_relay/engine.py) runs the four steps, [`plan.py`](src/horizon_relay/plan.py) validates the planner's task graph, [`budget.py`](src/horizon_relay/budget.py) counts cloud calls, [`events.py`](src/horizon_relay/events.py) defines progress events. [`relay.py`](src/horizon_relay/relay.py) is the public entry point, [`cli.py`](src/horizon_relay/cli.py) the command line, [`demo.py`](src/horizon_relay/demo.py) the scripted scenarios.

## Use it from Python

```python
from horizon_relay import Relay

relay = Relay.from_env()  # LOCAL_BASE_URL, LOCAL_MODEL, CLOUD_BASE_URL, CLOUD_MODEL, CLOUD_API_KEY, CLOUD_ENABLED

result = await relay.chat([{"role": "user", "content": "Compare SQLite and Postgres for a class project"}])
print(result.content)    # final answer (Markdown)
print(result.summary())  # "Relay: 4 local calls · 2 cloud calls · tokens: 3954."
result.metrics           # every call: tier, step, model, tokens, outcome, time

# Or answer a goal over your own named text sources instead of a chat:
result = await relay.run("List the action items", {"notes": open("meeting.txt").read()})
```

Pass `emit=` to receive progress as it happens. Each `RelayEvent` has `name` (e.g. `plan_created`, `task_invalid`, `run_completed`), a human-readable `message`, `done`, and details in `data`:

```python
async def show(event):
    print(event.seq, event.name, event.message)

await relay.chat(messages, emit=show)
```

Either tier can be any OpenAI-compatible Chat Completions server (llama.cpp, vLLM, a hosted API). Local requests also send `response_format` (JSON) and `chat_template_kwargs`, which llama.cpp and vLLM accept. Set the URLs and model names, or build `RelaySettings(Endpoint(...), Endpoint(...))` and call `Relay.from_settings(settings)`.

### Middle tier

Set `MID_BASE_URL` and `MID_MODEL` (and `MID_API_KEY` if needed) to add a model between the local and cloud ones, such as the 4B. The planner can then assign moderate work to `"mid"`, and failed subtasks climb one tier at a time: a local subtask tries local, local, mid, cloud.

### Decision graph

`RunGraph` turns a run's events into plain JSON: one lane per model; for each local model its answers, token probabilities, critic verdict, six signals and score against the threshold; then the planner's reason for each delegated subtask, the failed check behind each retry or escalation, and every model's self-reported confidence. `render_html()` inlines it into a self-contained page with no network requests:

```python
from horizon_relay.graph import RunGraph, render_html

graph = RunGraph()
result = await relay.chat(messages, emit=graph.aadd)
open("run.html", "w").write(render_html(graph.to_dict()))
```

The Open WebUI Pipe sends that page as a message embed after each step, so the graph builds live in the chat. The CLI writes it with `--graph run.html`. Confidence is shown as the model's judgment only; results are still accepted by checks alone.

## Change the routing

All decisions live in [`policy.py`](src/horizon_relay/policy.py). For each local model the relay records six signals, each 0–1:

| Signal | Captures | Measured as |
| --- | --- | --- |
| `self_uncertainty` | what the model says about itself | 1 − average self-reported confidence |
| `token_uncertainty` | how sure it was of the words it wrote | 1 − geometric-mean probability of the answer's tokens (logprobs); left out when unavailable |
| `inconsistency` | whether it answers the same twice | 1 − similarity of its two answers |
| `critic_risk` | what a second look found | severity from the critic (the next larger local model) |
| `task_complexity` | how hard the request looks | keyword/length heuristic |
| `explicit_escalation` | whether it asked for help | 1 if it set `needs_escalation` |

**Rules** run first and can force escalation: the model asked for help or gave no answer; the critic rejected the answer with severity ≥ 0.5; the 0.9B's token uncertainty ≥ 0.12. Otherwise the **score** (weighted mean of the available signals) is compared with the model's **threshold**: 0.50 for the 0.9B, 0.25 for the 4B.

```python
from horizon_relay import Relay, RoutingPolicy
from horizon_relay.policy import critic_fails, escalate_when_requested, token_uncertainty_at_least

policy = RoutingPolicy(
    escalation_threshold=0.5, tier_thresholds={"mid": 0.25},
    weights={"self_uncertainty": 0.25, "token_uncertainty": 0.15, "inconsistency": 0.25,
             "critic_risk": 0.25, "task_complexity": 0.10, "explicit_escalation": 0.10},
    rules=(escalate_when_requested, critic_fails(0.5), token_uncertainty_at_least(0.12)),
    skip_small_above=None,   # e.g. 0.3: send requests that look hard straight past the 0.9B
    cloud_mode="plan",       # or "direct"
)
relay = Relay.from_env(policy=policy)
```

A rule is any function taking a `Verdict` (tier, answers, critique, signals) and returning True (escalate), False (keep) or None (no opinion). Environment variables cover the common knobs: `ESCALATION_THRESHOLD`, `LOCAL_SAMPLES`, `RELAY_CROSS_CRITIC`, `RELAY_CLOUD_MODE`, `SKIP_SMALL_ABOVE`.

### Routing tests and tuning

[`fixtures/routing_tests.json`](src/horizon_relay/fixtures/routing_tests.json) holds 25 labeled requests: 7 for the 0.9B, 8 for the 4B, 7 for the cloud and 3 that force a path (0.9B → 4B, cloud plan-and-delegate, cloud direct). Each lists the models allowed to answer and, where possible, a pattern the answer must match.

- `horizon-relay eval [selection]` runs them and reports route and answer accuracy (`--graph-dir` saves each graph). In Open WebUI the **Horizon Relay Tests** model does the same with `/run` and saves one chat per test in the **Tests** folder.
- `horizon-relay tune` has both local models answer every test with escalation forced and the cloud off (no cloud calls), saves those profiles, and searches weights, thresholds and rules against the labels, penalizing any setting that would return a wrong local answer. `--reuse` re-searches saved profiles offline in seconds.

The current defaults came from that search: 20 of 22 tests routed as labeled, up from 12, with 2 wrong local answers instead of 6. Twenty-odd tests guide the settings; they do not validate them. Re-tune after changing models or prompts.

## Command line

```sh
python3 -m pip install -e './relay[live]'     # or run with PYTHONPATH=relay/src
horizon-relay ask "Compare SQLite and Postgres for a class project" --graph run.html
horizon-relay ask "List the action items" --source notes=meeting.txt --events
horizon-relay eval 4b                          # routing tests expected on the 4B
horizon-relay tune                             # profile the local models and search settings
horizon-relay demo --scenario escalate --graph run.html   # scripted; no model or network calls
```

`ask` and `eval` read the same environment variables as `Relay.from_env()`. Progress events (`--events`) and the call summary go to stderr, the answer to stdout; `--json` prints the full result. Inside the Docker stack: `docker compose -p horizon-relay exec open-webui horizon-relay ...`.

## Add a front end or provider

- **Front end**: call `Relay.chat` or `Relay.run`, show `RelayEvent`s however you like, and display `result.content`. The Open WebUI Pipe is a complete example.
- **Provider**: implement `async complete(tier, operation, payload) -> Reply` and a `simulated` attribute (see [`providers/__init__.py`](src/horizon_relay/providers/__init__.py)), then pass it to `Relay(provider)`. `SimulatedProvider` is a small reference.

## Tests

```sh
PYTHONPATH=relay/src python3 -m unittest discover -s relay/tests -v
PYTHONPATH=relay/src python3 -m unittest discover -s integrations/openwebui -v
```

The core needs only Python 3.11+. HTTP adapter tests run when `httpx` is installed (the `live` extra) and use mock transports, so they make no API calls. A boundary test fails if front-end code leaks into the package.
