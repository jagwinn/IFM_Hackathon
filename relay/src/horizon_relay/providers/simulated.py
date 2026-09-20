"""Deterministic scripted provider over two built-in bug reports; no network calls."""

import asyncio
import json
from importlib.resources import files

from . import Reply

SCENARIOS = {"standard", "trusted", "local", "repair", "escalate"}


def demo_sources():
    return json.loads(files("horizon_relay").joinpath("fixtures/bug_reports.json").read_text())


class SimulatedProvider:
    simulated = True
    models = {"local": "simulated-horizon-local", "cloud": "simulated-horizon-cloud"}

    def __init__(self, scenario="standard", delay=0):
        if scenario not in SCENARIOS:
            raise ValueError("Unknown demo scenario")
        if not 0 <= delay <= 2:
            raise ValueError("Demo delay must be between 0 and 2 seconds")
        self.scenario = scenario
        self.delay = delay

    @property
    def notice(self):
        """Announced at the start of each run so injected failures are never mistaken for real ones."""
        if self.scenario in ("repair", "escalate"):
            return "Demo intentionally injects an invalid worker quote"
        return None

    async def complete(self, tier, operation, payload):
        await asyncio.sleep(self.delay)
        model = f"simulated-horizon-{tier}"
        if operation == "solve":
            if self.scenario == "trusted":
                return Reply({"answer": "report_1 is a checkout failure after the SAVE10 coupon (error 500).",
                              "confidence": 0.93, "task_type": "summary", "difficulty": "low",
                              "needs_escalation": False, "reason": "The report states it directly."}, model)
            return Reply({"answer": "Report 1 may matter more, but I cannot rank all reports reliably.",
                          "confidence": 0.3, "task_type": "prioritization", "difficulty": "high",
                          "needs_escalation": True, "reason": "Several reports to compare and prioritize."}, model)
        if operation == "critique":
            return Reply({"passed": True, "severity": 0.05, "critique": "Matches the report."}, model)
        if operation == "answer":
            return Reply("### Direct cloud answer (scripted fixture)\n\nFix the coupon checkout failure first.", model)
        if operation == "attempt":
            if self.scenario == "local":
                return Reply({"report_id": "report_1", "quote": payload["sources"]["report_1"], "confidence": "high"}, model)
            return Reply({"needs_plan": True, "confidence": "low",
                          "reason": "Several reports to compare and prioritize."}, model)
        if operation == "plan":
            tasks = []
            for report_id in payload["sources"]:
                tasks.append({
                    "id": f"extract_{report_id}", "instruction": "Extract the report verbatim as evidence.",
                    "why": "Copying one report word for word is literal and checkable.",
                    "preferred_tier": "local", "input_refs": [f"sources.{report_id}"],
                    "depends_on": [], "result_type": "report_extraction",
                    "checks": ["required_fields", "source_quotes_match"],
                })
            extraction_ids = [t["id"] for t in tasks]
            tasks.append({
                "id": "group_reports", "instruction": "Group the extracted reports by symptom.",
                "why": "Sorting a few short reports into groups needs little reasoning.",
                "preferred_tier": "local", "input_refs": [f"results.{i}" for i in extraction_ids],
                "depends_on": extraction_ids, "result_type": "report_groups",
                "checks": ["required_fields", "report_ids_preserved"],
            })
            return Reply({"version": 2, "tasks": tasks, "final_instruction": "Prioritize fixes using the evidence."}, model)
        if operation == "work":
            task = payload["task"]
            failures = {"repair": 1, "escalate": 2}.get(self.scenario, 0)
            if task["id"] == "extract_report_1" and tier == "local" and payload["attempt"] <= failures:
                return Reply({"report_id": "report_1", "quote": "INJECTED INVALID QUOTE", "confidence": "low",
                              "concern": "Scripted fault: the quote is deliberately wrong."}, model)
            if task["result_type"] == "report_extraction":
                ref, text = next(iter(payload["inputs"].items()))
                return Reply({"report_id": ref.split(".", 1)[1], "quote": text, "confidence": "high"}, model)
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
        if operation == "direct":
            return Reply("Simulated reply; no model called.", model)
        raise ValueError("Unsupported simulated operation")
