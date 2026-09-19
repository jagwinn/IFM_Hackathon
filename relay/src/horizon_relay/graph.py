"""A run's decision graph, built from its events.

RunGraph answers, for every step of a run: which model handled it, why that model, what each attempt
returned, what the model judged about its own work, and where work was delegated down or escalated up.
It is plain JSON (to_dict) with no display code; ui/graph.html draws it and render_html() inlines one.

    graph = RunGraph()
    result = await relay.chat(messages, emit=graph.aadd)
    html = render_html(graph.to_dict())
"""

from importlib.resources import files
import json

from .events import RelayEvent

POLICY = "Routing policy"


def _title(task_id: str) -> str:
    return task_id.replace("_", " ").strip().capitalize() or task_id


class RunGraph:
    def __init__(self):
        self.run_id = None
        self.status = "running"  # running, completed, failed, cancelled
        self.message = ""
        self.simulated = False
        self.goal = ""
        self.lanes: list[dict] = []  # smallest model first
        self.nodes: dict[str, dict] = {}  # insertion order is display order within a lane
        self.edges: list[dict] = []
        self.task_lane: dict[str, str] = {}  # task_id -> tier the planner assigned
        self.accepted: dict[str, str] = {}  # task_id -> node that produced the accepted result
        self.final_instruction = ""  # the planner's instruction for the final answer
        self.metrics = None
        self.route = None  # the tier whose answer was returned, once the run completes

    async def aadd(self, event: RelayEvent) -> None:
        """Async form of add(), usable directly as a relay `emit` callback."""
        self.add(event)

    def add(self, event: RelayEvent) -> None:
        self.run_id, self.simulated = event.run_id, event.simulated
        handler = getattr(self, "_on_" + event.name, None)
        if handler:
            handler(event.data, event)

    # Nodes ---------------------------------------------------------------

    def _model(self, tier):
        return next((lane["model"] for lane in self.lanes if lane["id"] == tier), tier)

    def _node(self, node_id, **fields):
        if node_id not in self.nodes:
            self.nodes[node_id] = {"id": node_id, "status": "queued", "attempts": [], "judgment": None,
                                   "result": None, "why": "", "by": "", "instruction": "", "depends_on": [],
                                   "task_id": None, **fields}
        return self.nodes[node_id]

    def _task_node(self, task_id, tier):
        """The original node in the planner's lane, or an escalated retry node in a larger model's lane."""
        if tier == self.task_lane.get(task_id, tier):
            return self._node(task_id)
        original = self.nodes.get(task_id, {})
        return self._node(f"{task_id}@{tier}", kind="retry", lane=tier, round=original.get("round", 1),
                          title=original.get("title", _title(task_id)), task_id=task_id,
                          instruction=original.get("instruction", ""), inputs=original.get("inputs", []),
                          result_type=original.get("result_type"), by=POLICY,
                          why=("Its checks failed on the smaller model, so the routing policy gives only this "
                               f"subtask a try on {self._model(tier)}. The rest of the plan is unaffected."))

    def _attempt(self, node, n):
        for attempt in node["attempts"]:
            if attempt["n"] == n:
                return attempt
        attempt = {"n": n, "lane": node["lane"], "model": self._model(node["lane"]), "outcome": "running",
                   "check": None, "ms": None, "tokens": None, "judgment": None}
        node["attempts"].append(attempt)
        return attempt

    def _edge(self, source, target, kind, label=None):
        edge = {"from": source, "to": target, "type": kind, "label": label}
        if edge not in self.edges:
            self.edges.append(edge)

    # Event handlers --------------------------------------------------------

    def _on_run_started(self, d, e):
        self.goal = d.get("goal", "")
        self.lanes = [{"id": t["id"], "model": t["model"], "location": t.get("location", "")} for t in d.get("tiers", [])]

    def _on_local_attempt(self, d, e):
        """/local-extract: an exact copy of one source, checked before any model is trusted."""
        node = self._node("triage", kind="triage", lane="local", round=0, title="Exact extraction", status="running",
                          instruction=self.goal, by=POLICY,
                          why="You asked for an exact copy (/local-extract), which can be checked word for word locally.")
        self._attempt(node, 1)

    def _on_call_finished(self, d, e):
        op, tier = d["operation"], d["tier"]
        if op == "attempt":
            node = self.nodes.get("triage")
        elif op in ("solve", "critique"):
            node = self._judge(d.get("for_tier") or tier)
        elif op == "answer":
            node = self._answer_node(direct=True)
        elif op == "plan":
            node = self._plan_node()
        elif op == "synthesize":
            node = self._answer_node()
        elif op == "work" and d.get("task_id"):
            node = self._task_node(d["task_id"], tier)
        else:
            return
        if node is None:
            return
        attempt = self._attempt(node, d.get("attempt") or 1)
        tokens = None
        if d.get("prompt_tokens") is not None and d.get("completion_tokens") is not None:
            tokens = d["prompt_tokens"] + d["completion_tokens"]
        attempt.update(lane=tier, model=d.get("model") or attempt["model"], ms=d.get("elapsed_ms"),
                       tokens=tokens, judgment=d.get("judgment"))
        if d.get("judgment"):
            node["judgment"] = d["judgment"]
        if d.get("outcome") == "failed":
            attempt.update(outcome="error", check="The model call failed (connection, timeout or invalid reply).")
            node["status"] = "failed"
        elif op not in ("work", "solve", "critique"):
            attempt["outcome"] = "passed"  # plan_invalid overrides for a rejected plan

    def _on_extract_failed(self, d, e):
        self.nodes["triage"].update(status="escalated", result="Not an exact copy of the source")

    def _judge(self, tier):
        return self.nodes.get(f"judge:{tier}")

    def _on_local_solve(self, d, e):
        tier = d["tier"]
        previous = [n for n in self.nodes.values() if n.get("kind") == "judge"]
        node = self._node(f"judge:{tier}", kind="judge", lane=tier, round=0, title="Answer locally", status="running",
                          instruction=self.goal, by=POLICY, samples=[], critique=None, signals=None, weights=None,
                          threshold=None, decision=None, reason="",
                          why=("Every request starts on the smallest model. It answers and rates its own confidence, a "
                               "critic checks the answer, and the answer is trusted only if the combined risk score "
                               "stays under the threshold." if not previous else
                               "The smaller model's answer was not trusted, so the next model up answers and is scored "
                               "the same way."))
        source = previous[-1] if previous else self.nodes.get("triage")
        if source:
            label = source.get("reason") if source.get("kind") == "judge" else "Not an exact copy"
            self._edge(source["id"], node["id"], "escalated", label)

    def _on_local_sample(self, d, e):
        node = self._judge(d["tier"])
        sample = d["assessment"]
        node["samples"].append(sample)
        attempt = self._attempt(node, d["attempt"])
        attempt.update(label=f"Answer {d['attempt']}", confidence=sample["confidence"], token_prob=sample.get("token_prob"))
        if sample["needs_escalation"]:
            attempt.update(outcome="failed", check=("Asked for a larger model: " if sample["structured"] else "")
                           + (sample["reason"] or "no reason given"))
        else:
            attempt["outcome"] = "passed"
        level = "low" if sample["confidence"] < 0.5 else "medium" if sample["confidence"] < 0.8 else "high"
        node["judgment"] = {"confidence": level, "score": sample["confidence"], "note": sample["reason"]}

    def _on_local_skipped(self, d, e):
        tier = d["tier"]
        self._node(f"judge:{tier}", kind="judge", lane=tier, round=0, title="Answer locally", status="skipped",
                   instruction=self.goal, by=POLICY, samples=[], critique=None, signals=None, weights=None,
                   threshold=None, decision="skip", reason=f'Complexity {d["complexity"]:.2f} ≥ {d["threshold"]:.2f}',
                   why=("This request looks hard (task complexity at or above the policy's skip_small_above), so the "
                        "smallest model is not asked at all."))

    def _on_local_critique(self, d, e):
        node = self._judge(d["tier"])
        critic = d.get("critic_tier") or d["tier"]
        label = "Critic" if critic == d["tier"] else f"Critic ({self._model(critic).split('/')[-1]})"
        self._attempt(node, len(node["samples"]) + 1).update(label=label, lane=critic, model=self._model(critic))

    def _on_local_verdict(self, d, e):
        node = self._judge(d["tier"])
        verdict = d["verdict"]
        node.update(critique=verdict["critique"], signals=verdict["signals"], threshold=verdict["threshold"],
                    weights=d.get("weights"), decision="escalate" if verdict["escalate"] else "trust",
                    reason=verdict["reason"], status="escalated" if verdict["escalate"] else "passed",
                    result=verdict["samples"][0]["answer"])
        critique = verdict["critique"]
        if not critique["skipped"]:
            critic = self._attempt(node, len(node["samples"]) + 1)
            critic.update(outcome="passed" if critique["passed"] else "failed",
                          check=None if critique["passed"] else f'Severity {critique["severity"]:.2f}: {critique["critique"]}')

    def _on_local_accepted(self, d, e):
        judge = self._judge(d.get("tier"))
        if judge:
            judge["status"] = "passed"
        elif "triage" in self.nodes:
            self.nodes["triage"].update(status="answered", result=e.message)

    def _on_escalated(self, d, e):
        source = next((n for n in reversed(list(self.nodes.values())) if n.get("kind") in ("judge", "triage")), None)
        target = self._answer_node(direct=True) if d.get("mode") == "direct" else self._plan_node()
        target["status"] = "running"
        if source:
            self._edge(source["id"], target["id"], "escalated", source.get("reason") or "Escalated")

    def _on_answering(self, d, e):
        self._answer_node(direct=True)["status"] = "running"

    def _plan_node(self):
        return self._node("plan", kind="plan", lane="cloud", round=0, title="Plan the work", by=POLICY,
                          instruction="Split the request into subtasks and give each to the smallest model that can handle it.",
                          why=("No local answer was trusted, so the large model breaks the request down and delegates "
                               "each piece to the smallest model expected to handle it (cloud_mode=plan)."))

    def _answer_node(self, direct=False):
        last = max((n["round"] for n in self.nodes.values() if n.get("kind") in ("task", "retry")), default=0)
        if direct:
            return self._node("answer", kind="answer", lane="cloud", round=last, title="Answer in the cloud", by=POLICY,
                              instruction="Solve the original request, using the local attempts only as clues.",
                              why=("No local answer was trusted, and the policy sends such requests straight to the "
                                   "large model (cloud_mode=direct)."))
        return self._node("answer", kind="answer", lane="cloud", round=last + 1, title="Write the answer", by=POLICY,
                          instruction=self.final_instruction or "Answer the original request from the checked subtask results.",
                          inputs=[f"Result of {n['title']}" for n in self.nodes.values() if n.get("kind") == "task"]
                                 + ["The original request"],
                          why=("Only the large model writes the final answer. Subtask results are format-checked, "
                               "not fact-checked, so it reviews their reasoning here."))

    def _on_plan_invalid(self, d, e):
        attempt = self._attempt(self._plan_node(), d.get("attempt") or 1)
        attempt.update(outcome="failed", check=d.get("error"))

    def _on_plan_created(self, d, e):
        tasks = d.get("tasks", [])
        rounds: dict[str, int] = {}
        pending = list(tasks)
        while pending:  # dependency depth; the plan is already validated as acyclic
            for task in list(pending):
                if all(dep in rounds for dep in task["depends_on"]):
                    rounds[task["id"]] = 1 + max((rounds[dep] for dep in task["depends_on"]), default=0)
                    pending.remove(task)
        plan = self._plan_node()
        planner = self._model("cloud")
        delegated = sum(t["preferred_tier"] != "cloud" for t in tasks)
        self.final_instruction = d.get("final_instruction") or ""
        plan.update(status="passed", summary=f"{len(tasks)} subtasks · {delegated} delegated",
                    subtasks=[{"title": _title(t["id"]), "model": self._model(t["preferred_tier"]), "lane": t["preferred_tier"],
                               "instruction": t["instruction"], "why": t.get("why") or ""} for t in tasks],
                    result=f"Final answer instruction: {self.final_instruction}" if self.final_instruction else None)
        titles = {t["id"]: _title(t["id"]) for t in tasks}

        def describe(ref):
            kind, _, key = ref.partition(".")
            if kind == "results":
                return f"Result of {titles.get(key, key)}"
            return "The conversation" if key == "conversation" else f"Source “{key}”"

        for task in tasks:
            self.task_lane[task["id"]] = task["preferred_tier"]
            self._node(task["id"], kind="task", lane=task["preferred_tier"], round=rounds[task["id"]],
                       title=_title(task["id"]), task_id=task["id"], instruction=task["instruction"],
                       inputs=[describe(r) for r in task["input_refs"]], result_type=task["result_type"],
                       depends_on=list(task["depends_on"]), why=task.get("why") or "",
                       by=f"Assigned by {planner} in the plan")
            self._edge("plan", task["id"], "delegated" if task["preferred_tier"] != "cloud" else "assigned")
            for dep in task["depends_on"]:
                self._edge(dep, task["id"], "result")

    def _on_task_escalated(self, d, e):
        source = self._task_node(d["task_id"], d["from_tier"])
        target = self._task_node(d["task_id"], d["tier"])
        source["status"] = "escalated"
        failures = sum(a["outcome"] == "failed" for a in source["attempts"])
        label = f"{failures} failed check{'s' if failures != 1 else ''} · escalated" if failures else "Escalated"
        self._edge(source["id"], target["id"], "escalated", label)
        target["escalated_from"] = source["id"]
        target["escalation_reason"] = d.get("reason")

    def _on_task_started(self, d, e):
        node = self._task_node(d["task_id"], d["tier"])
        node["status"] = "running"
        self._attempt(node, d["attempt"]).update(lane=d["tier"], model=d.get("model") or self._model(d["tier"]))

    def _on_task_invalid(self, d, e):
        node = self._task_node(d["task_id"], d["tier"])
        self._attempt(node, d["attempt"]).update(outcome="failed", check=d.get("error"), judgment=d.get("judgment"))
        node["status"] = "failed"
        if d.get("judgment"):
            node["judgment"] = d["judgment"]

    def _on_task_completed(self, d, e):
        node = self._task_node(d["task_id"], d["tier"])
        self._attempt(node, d["attempt"]).update(outcome="passed", judgment=d.get("judgment"))
        node.update(status="passed", result=d.get("result"))
        if d.get("judgment"):
            node["judgment"] = d["judgment"]
        self.accepted[d["task_id"]] = node["id"]

    def _on_synthesizing(self, d, e):
        answer = self._answer_node()
        answer["status"] = "running"
        needed = {dep for n in self.nodes.values() if n.get("kind") == "task" for dep in n["depends_on"]}
        for task_id in self.task_lane:
            if task_id not in needed:
                self._edge(task_id, "answer", "result")

    def _on_run_completed(self, d, e):
        self.status, self.message, self.metrics = "completed", e.message, d.get("metrics")
        self.route = (self.metrics or {}).get("route")
        if "answer" in self.nodes:
            self.nodes["answer"].update(status="passed", result="Shown below the graph.")

    def _on_run_failed(self, d, e):
        self._stop("failed", e.message, d.get("metrics"))

    def _on_run_cancelled(self, d, e):
        self._stop("cancelled", e.message, d.get("metrics"))

    def _stop(self, status, message, metrics):
        self.status, self.message, self.metrics = status, message, metrics
        for node in self.nodes.values():
            if node["status"] in ("running", "queued"):
                node["status"] = "stopped"
            for attempt in node["attempts"]:
                if attempt["outcome"] == "running":
                    attempt["outcome"] = "stopped"

    # Output ----------------------------------------------------------------

    def to_dict(self) -> dict:
        edges = []
        for edge in self.edges:
            source = self.accepted.get(edge["from"], edge["from"]) if edge["type"] == "result" else edge["from"]
            if source in self.nodes and edge["to"] in self.nodes:
                edges.append({**edge, "from": source})
        return {"version": 1, "run_id": self.run_id, "status": self.status, "message": self.message,
                "simulated": self.simulated, "goal": self.goal, "lanes": self.lanes, "route": self.route,
                "nodes": list(self.nodes.values()), "edges": edges, "metrics": self.metrics}


def render_html(graph: dict) -> str:
    """The standalone graph page (ui/graph.html) with `graph` inlined. Safe to use as an iframe srcdoc."""
    template = files("horizon_relay").joinpath("ui/graph.html").read_text()
    data = json.dumps(graph, ensure_ascii=False).replace("</", "<\\/")
    return template.replace("/*RELAY_GRAPH*/null", data, 1)
