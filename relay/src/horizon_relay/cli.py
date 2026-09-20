"""Command line: `horizon-relay ask "question"` (real models) and `horizon-relay demo` (scripted)."""

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .demo import SCENARIOS, run_demo
from .errors import RelayError
from .graph import RunGraph, render_html


def _emitter(args):
    """Progress callback for --events (JSONL on stderr) and --graph (decision graph HTML file)."""
    args.graph_data = RunGraph() if args.graph else None
    if not args.events and not args.graph:
        return None
    async def emit(event):
        if args.events:
            print(json.dumps(event.to_dict()), file=sys.stderr)
        if args.graph_data:
            args.graph_data.add(event)
    return emit


def _write_graph(args):
    if args.graph_data:
        Path(args.graph).write_text(render_html(args.graph_data.to_dict()))
        print(f"Decision graph written to {args.graph}", file=sys.stderr)


def _print(result, as_json):
    if as_json:
        print(json.dumps(asdict(result), indent=2))
    else:
        print(result.content)
        print("\n" + result.summary(), file=sys.stderr)


async def ask(args):
    from .relay import Relay
    sources = {}
    for item in args.source or ():
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"--source expects NAME=FILE, got {item!r}")
        sources[name] = Path(path).read_text()
    relay = Relay.from_env()
    emit = _emitter(args)
    try:
        if sources:
            result = await relay.run(args.question, sources, local_extract=args.local_extract, emit=emit)
        else:
            result = await relay.chat([{"role": "user", "content": args.question}], emit=emit)
    finally:
        _write_graph(args)
    _print(result, args.json)


async def evaluate(args):
    from .evaluation import load_cases, run_case, select_cases, summarize
    from .graph import render_html
    from .relay import Relay
    relay = Relay.from_env()
    cases = select_cases(load_cases(args.cases), args.select)
    if not cases:
        raise ValueError("No test case matches that selection")
    if args.graph_dir:
        Path(args.graph_dir).mkdir(parents=True, exist_ok=True)
    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.name} (expect {' or '.join(case.expected_route)})", file=sys.stderr)
        result = await run_case(relay, case)
        results.append(result)
        scores = " ".join(f"{t}={s:.2f}" for t, s in result.scores.items())
        answer = "" if result.correct is None else " answer " + ("correct" if result.correct else "WRONG")
        if result.case.expects:
            answer += f" delegated {result.local_subtasks}" + (f" ({result.parallel_ms}ms parallel)" if result.parallel_ms else "")
            answer += "" if not result.unmet else " UNMET: " + "; ".join(result.unmet)
        print(f"    -> {result.route} {'OK' if result.matched else 'MISMATCH'}{answer} {scores} {result.elapsed_s}s"
              + (f" error: {result.error}" if result.error else ""), file=sys.stderr)
        if args.graph_dir:
            slug = "".join(ch if ch.isalnum() else "-" for ch in case.name.lower()).strip("-")
            (Path(args.graph_dir) / f"{i:02d}-{slug}.html").write_text(render_html(result.graph))
    summary = summarize(results)
    if args.json:
        print(json.dumps({"summary": summary, "results": [{**r.to_dict(), "graph": None} for r in results]}, indent=2, default=str))
    else:
        print(json.dumps(summary, indent=2))


async def tune(args):
    from .evaluation import load_cases
    from .relay import Relay
    from .tuning import evaluate, profile, search
    path = Path(args.profiles)
    if args.reuse and path.exists():
        profiles = json.loads(path.read_text())
    else:
        cases = load_cases(args.cases)
        profiles = await profile(Relay.from_env(), cases,
                                 progress=lambda i, c: print(f"[profile {i + 1}] {c.name}", file=sys.stderr))
        path.write_text(json.dumps(profiles, indent=2))
        print(f"Profiles saved to {path}", file=sys.stderr)
    policy = Relay.from_env().engine.policy
    rule_limit = lambda name, attr: next((getattr(r, attr) for r in policy.rules if r.__name__ == name), None)
    current = {"weights": dict(policy.weights), "thresholds": {t: policy.threshold(t) for t in ("local", "mid")},
               "critic_min": rule_limit("escalate_when_critic_rejects", "min_severity"),
               "skip_small_above": policy.skip_small_above,
               "token_max": rule_limit("escalate_when_tokens_are_uncertain", "limit"),
               "token_min_length": rule_limit("escalate_when_tokens_are_uncertain", "min_tokens") or 1}
    ranked = search(profiles)
    best_params, best = ranked[0]
    report = {"current": {"params": current, "result": evaluate(current, profiles)},
              "best": {"params": best_params, "result": best},
              "runners_up": [{"params": p, "result": {k: v for k, v in r.items() if k != "routes"}} for p, r in ranked[1:6]],
              "cases": [{"name": p["name"], "expected": p["expected"], "current": c, "best": b}
                        for p, c, b in zip(profiles, evaluate(current, profiles)["routes"], best["routes"])]}
    print(json.dumps(report, indent=2))


async def demo(args):
    emit = _emitter(args)
    try:
        result = await run_demo(args.scenario, delay=args.delay, cloud_enabled=not args.no_cloud, emit=emit)
    finally:
        _write_graph(args)
    _print(result, args.json)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="horizon-relay", description="Local-first relay between a small local model and a large cloud model")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("ask", help="Answer a question with the real models configured by environment variables")
    p.add_argument("question")
    p.add_argument("--source", action="append", metavar="NAME=FILE", help="Answer using this text file as a named source (repeatable)")
    p.add_argument("--local-extract", action="store_true", help="With one --source: copy it exactly on the local model")
    p.set_defaults(run=ask)

    p = commands.add_parser("eval", help="Run the labeled routing tests against the real models and report route accuracy")
    p.add_argument("select", nargs="?", default="all", help='"all", positions like "1 3 5", a difficulty, category, expected route, or name words')
    p.add_argument("--cases", metavar="FILE", help="A JSON test set (default: the built-in routing_tests.json)")
    p.add_argument("--graph-dir", metavar="DIR", help="Write each test's decision graph as an HTML page")
    p.add_argument("--json", action="store_true", help="Print every result as JSON")
    p.set_defaults(run=evaluate)

    p = commands.add_parser("tune", help="Profile the local models on the routing tests (no cloud calls) and search routing settings")
    p.add_argument("--profiles", default="relay-profiles.json", metavar="FILE", help="Where profiles are saved (default: relay-profiles.json)")
    p.add_argument("--reuse", action="store_true", help="Reuse saved profiles instead of calling the models again")
    p.add_argument("--cases", metavar="FILE", help="A JSON test set (default: the built-in routing_tests.json)")
    p.set_defaults(run=tune)

    p = commands.add_parser("demo", help="Run a scripted scenario; no model or network calls")
    p.add_argument("--scenario", choices=sorted(SCENARIOS), default="standard")
    p.add_argument("--delay", type=float, default=0, help="Seconds per simulated call")
    p.add_argument("--no-cloud", action="store_true")
    p.set_defaults(run=demo)

    for name, p in commands.choices.items():
        if name in ("eval", "tune"):
            continue
        p.add_argument("--events", action="store_true", help="Write progress events as JSONL to stderr")
        p.add_argument("--json", action="store_true", help="Print the full result (content, results, metrics) as JSON")
        p.add_argument("--graph", metavar="FILE", help="Write the run's decision graph as a standalone HTML page")

    args = parser.parse_args(argv)
    try:
        asyncio.run(args.run(args))
    except (RelayError, ValueError, OSError) as exc:
        parser.exit(1, f"Relay stopped: {exc}\n")
