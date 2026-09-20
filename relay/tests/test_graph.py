import json
import unittest

from horizon_relay import Relay, Reply, RoutingPolicy, SimulatedProvider
from horizon_relay.demo import run_demo
from horizon_relay.graph import RunGraph, render_html


class ThreeTiers(SimulatedProvider):
    """local fails twice, mid fails once, cloud passes: the full 0.9B -> 4B -> cloud ladder."""
    models = {"local": "IFM/K2-Horizon-0.9B", "mid": "IFM/K2-Horizon-4B", "cloud": "IFM/K2-Horizon-375B-A23B"}

    async def locations(self):
        return {"local": "RTX 3070", "mid": "RTX 3070", "cloud": "IFM API"}

    async def complete(self, tier, operation, payload):
        if operation == "plan":
            return Reply({"version": 2, "final_instruction": "Answer", "tasks": [
                {"id": "extract", "instruction": "List constraints", "preferred_tier": "local",
                 "why": "Literal copying", "input_refs": ["sources.conversation"], "depends_on": [],
                 "result_type": "task_answer", "checks": ["required_fields"]},
                {"id": "compare", "instruction": "Compare options", "preferred_tier": "mid",
                 "why": "Moderate reasoning", "input_refs": ["results.extract"], "depends_on": ["extract"],
                 "result_type": "task_answer", "checks": ["required_fields"]},
            ]}, "large")
        if operation == "work":
            if payload["task"]["id"] == "extract" and tier in ("local", "mid"):
                if tier == "local":
                    return Reply({"answer": "", "confidence": "low", "concern": "Unsure what counts"}, "small")
                return Reply("not json", "mid")
            return Reply({"answer": f"{payload['task']['id']} done", "confidence": "medium", "concern": ""}, tier)
        if operation == "synthesize":
            return Reply("Final", "large")
        return await super().complete(tier, operation, payload)


class GraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_escalation_ladder_and_delegation(self):
        graph = RunGraph()
        result = await Relay(ThreeTiers()).chat([{"role": "user", "content": "Compare two options"}], emit=graph.aadd)
        g = graph.to_dict()
        nodes = {n["id"]: n for n in g["nodes"]}
        self.assertEqual([lane["id"] for lane in g["lanes"]], ["local", "mid", "cloud"])
        self.assertEqual([lane["location"] for lane in g["lanes"]], ["RTX 3070", "RTX 3070", "IFM API"])
        self.assertEqual(g["status"], "completed")
        # The 0.9B and then the 4B answered, judged themselves and were not trusted.
        for tier in ("local", "mid"):
            judge = nodes[f"judge:{tier}"]
            self.assertEqual((judge["status"], judge["decision"]), ("escalated", "escalate"))
            self.assertEqual(judge["judgment"]["confidence"], "low")
            self.assertEqual(set(judge["signals"]), {"self_uncertainty", "token_uncertainty", "inconsistency", "critic_risk",
                                                     "task_complexity", "explicit_escalation", "combined_score"})
            self.assertEqual(judge["weights"]["critic_risk"], 0.25)
            self.assertEqual(judge["threshold"], {"local": 0.5, "mid": 0.25}[tier])
        # The planner delegated down with reasons.
        self.assertEqual(nodes["extract"]["lane"], "local")
        self.assertEqual(nodes["extract"]["why"], "Literal copying")
        delegated = {(e["from"], e["to"]) for e in g["edges"] if e["type"] == "delegated"}
        self.assertEqual(delegated, {("plan", "extract"), ("plan", "compare")})
        # 0.9B failed twice (with its judgment), 4B once, then the cloud passed.
        self.assertEqual([a["outcome"] for a in nodes["extract"]["attempts"]], ["failed", "failed"])
        self.assertEqual(nodes["extract"]["attempts"][0]["judgment"]["note"], "Unsure what counts")
        self.assertIn("nonempty answer", nodes["extract"]["attempts"][0]["check"])
        self.assertEqual(nodes["extract@mid"]["status"], "escalated")
        self.assertEqual(nodes["extract@cloud"]["status"], "passed")
        escalations = [(e["from"], e["to"]) for e in g["edges"] if e["type"] == "escalated"]
        self.assertEqual(escalations, [("judge:local", "judge:mid"), ("judge:mid", "plan"),
                                       ("extract", "extract@mid"), ("extract@mid", "extract@cloud")])
        # The next subtask receives the accepted (cloud) result, and the answer reads the leaf task.
        results = {(e["from"], e["to"]) for e in g["edges"] if e["type"] == "result"}
        self.assertEqual(results, {("extract@cloud", "compare"), ("compare", "answer")})
        self.assertEqual(nodes["compare"]["round"], 2)
        # Every subtask shows what it was asked, with which inputs, and what it must return.
        self.assertEqual((nodes["compare"]["instruction"], nodes["compare"]["inputs"], nodes["compare"]["result_type"]),
                         ("Compare options", ["Result of Extract"], "task_answer"))
        self.assertEqual(nodes["extract@cloud"]["instruction"], "List constraints")
        self.assertEqual([t["instruction"] for t in nodes["plan"]["subtasks"]], ["List constraints", "Compare options"])
        self.assertEqual(nodes["answer"]["instruction"], "Answer")
        self.assertEqual(g["route"], "cloud")
        self.assertEqual(result.metrics["local_calls"], 6)  # 0.9B and 4B answers, 2 local, 1 mid, 1 mid compare

    async def test_exact_extraction_graph(self):
        graph = RunGraph()
        await run_demo("local", emit=graph.aadd)
        g = graph.to_dict()
        self.assertEqual([n["id"] for n in g["nodes"]], ["triage"])
        self.assertEqual(g["nodes"][0]["status"], "answered")
        self.assertTrue(g["simulated"])

    async def test_trusted_local_answer_graph(self):
        graph = RunGraph()
        await run_demo("trusted", emit=graph.aadd)
        g = graph.to_dict()
        judge = g["nodes"][0]
        self.assertEqual([n["id"] for n in g["nodes"]], ["judge:local"])
        self.assertEqual((judge["status"], g["route"]), ("passed", "local"))
        self.assertEqual([a["label"] for a in judge["attempts"]], ["Answer 1", "Answer 2", "Critic"])
        self.assertTrue(all(a["outcome"] == "passed" for a in judge["attempts"]))

    async def test_direct_cloud_graph(self):
        graph = RunGraph()
        await Relay(SimulatedProvider(), policy=RoutingPolicy(cloud_mode="direct")).run("Goal", {"a": "text"}, emit=graph.aadd)
        g = graph.to_dict()
        self.assertEqual([n["id"] for n in g["nodes"]], ["judge:local", "answer"])
        self.assertEqual(g["edges"], [{"from": "judge:local", "to": "answer", "type": "escalated",
                                       "label": "Rule: escalate when requested"}])
        self.assertEqual(g["nodes"][1]["round"], 0)

    async def test_failed_run_stops_nodes(self):
        graph = RunGraph()
        relay = Relay(SimulatedProvider("escalate"), policy=RoutingPolicy(escalate_failed_local_tasks=False))
        with self.assertRaises(Exception):
            await relay.run("Goal", {"report_1": "a", "report_2": "b"}, emit=graph.aadd)
        g = graph.to_dict()
        self.assertEqual(g["status"], "failed")
        self.assertFalse(any(n["status"] in ("running", "queued") for n in g["nodes"]))

    async def test_render_html_inlines_escaped_graph(self):
        graph = RunGraph()
        await run_demo("escalate", emit=graph.aadd)
        data = graph.to_dict()
        data["goal"] = "</script><script>alert(1)</script>"
        html = render_html(data)
        self.assertNotIn("/*RELAY_GRAPH*/null", html)
        self.assertNotIn("</script><script>alert", html)
        start = html.index("const G = ") + len("const G = ")
        end = html.index(";\n  if (!G) return;")
        self.assertEqual(json.loads(html[start:end].replace("<\\/", "</"))["goal"], data["goal"])


if __name__ == "__main__":
    unittest.main()
