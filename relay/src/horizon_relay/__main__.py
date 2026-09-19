import argparse
import asyncio
import json
import sys

from .config import Limits
from .engine import RelayEngine
from .providers import SCENARIOS, SimulatedProvider, demo_sources
from .schemas import RelayError


async def demo(args):
    async def emit(event):
        if args.events:
            print(json.dumps(event), file=sys.stderr)
    sources = demo_sources()
    if args.scenario == "local":
        sources = {"report_1": sources["report_1"]}
    result = await RelayEngine(SimulatedProvider(args.scenario, args.delay),
                               Limits(cloud_enabled=not args.no_cloud)).run(
        "Prioritize fixes for the supplied bug reports" if args.scenario != "local" else "Extract report 1 verbatim",
        sources, accept_local=args.scenario == "local", emit=emit,
    )
    print(result.content)
    print("\n" + json.dumps(result.metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Horizon Relay — explicitly simulated orchestration demo")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="standard")
    parser.add_argument("--delay", type=float, default=0)
    parser.add_argument("--events", action="store_true", help="Write status events as JSONL to stderr")
    parser.add_argument("--no-cloud", action="store_true")
    args = parser.parse_args()
    try:
        asyncio.run(demo(args))
    except (RelayError, ValueError) as exc:
        parser.exit(1, f"Relay stopped: {exc}\n")


if __name__ == "__main__":
    main()
