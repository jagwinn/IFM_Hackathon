"""Provider protocol and explicit deterministic simulation; no network calls."""

import asyncio
import json
from importlib.resources import files
from typing import Protocol

from .schemas import Reply

SCENARIOS = {"standard", "local", "repair", "escalate"}


def demo_sources():
    return json.loads(files("horizon_relay").joinpath("fixtures/bug_reports.json").read_text())


class Provider(Protocol):
    simulated: bool

    async def complete(self, tier: str, operation: str, payload: dict) -> Reply: ...


class SimulatedProvider:
    simulated = True

    def __init__(self, scenario="standard", delay=0):
        if scenario not in SCENARIOS:
            raise ValueError("Unknown demo scenario")
        if not 0 <= delay <= 2:
            raise ValueError("Demo delay must be between 0 and 2 seconds")
        self.scenario = scenario
        self.delay = delay

    async def complete(self, tier, operation, payload):
        await asyncio.sleep(self.delay)
        model = f"simulated-horizon-{tier}"
        if operation == "attempt":
            if self.scenario == "local":
                return Reply({"report_id": "report_1", "quote": payload["sources"]["report_1"]}, model)
            return Reply({"needs_plan": True}, model)
        if operation == "plan":
            tasks = []
            for report_id in payload["sources"]:
                tasks.append({
                    "id": f"extract_{report_id}", "instruction": "Extract the report verbatim as evidence.",
                    "preferred_tier": "local", "input_refs": [f"sources.{report_id}"],
                    "depends_on": [], "result_type": "report_extraction",
                    "checks": ["required_fields", "source_quotes_match"],
                })
            extraction_ids = [t["id"] for t in tasks]
            tasks.append({
                "id": "group_reports", "instruction": "Group the extracted reports by symptom.",
                "preferred_tier": "local", "input_refs": [f"results.{i}" for i in extraction_ids],
                "depends_on": extraction_ids, "result_type": "report_groups",
                "checks": ["required_fields", "report_ids_preserved"],
            })
            return Reply({"version": 1, "tasks": tasks, "final_instruction": "Prioritize fixes using the evidence."}, model)
        if operation == "work":
            task = payload["task"]
            failures = {"repair": 1, "escalate": 2}.get(self.scenario, 0)
            if task["id"] == "extract_report_1" and tier == "local" and payload["attempt"] <= failures:
                return Reply({"report_id": "report_1", "quote": "INJECTED INVALID QUOTE"}, model)
            if task["result_type"] == "report_extraction":
                ref, text = next(iter(payload["inputs"].items()))
                return Reply({"report_id": ref.split(".", 1)[1], "quote": text}, model)
            return Reply({"groups": [
                {"label": "Checkout failure" if v["report_id"] == "report_1" else "Cart display",
                 "report_ids": [v["report_id"]]}
                for v in payload["inputs"].values()
            ]}, model)
        if operation == "synthesize":
            evidence = [v for v in payload["results"].values() if "quote" in v]
            content = "### Engineering plan (scripted fixture)\n\n"
            content += "1. Investigate the coupon checkout failure first; it prevents order creation.\n"
            content += "2. Fix cart subtotal refresh and add a regression check for item removal.\n\n"
            content += "Evidence checked against the supplied fixture:\n\n"
            content += "\n\n".join(f'**{v["report_id"]}**: {v["quote"]}' for v in evidence)
            return Reply(content, model)
        if operation == "background":
            task = payload.get("task", "")
            if "title" in task:
                value = json.dumps({"title": "Horizon Relay simulated demo"})
            elif "tag" in task:
                value = json.dumps({"tags": ["Demo"]})
            elif "follow" in task:
                value = json.dumps({"follow_ups": []})
            else:
                value = "Simulated background task; no model called."
            return Reply(value, model)
        raise ValueError("Unsupported simulated operation")
