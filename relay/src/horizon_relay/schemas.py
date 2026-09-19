from dataclasses import dataclass
import re
from typing import Any


class RelayError(Exception):
    """An actionable, user-safe orchestration error."""


class ValidationError(RelayError):
    pass


class BudgetError(RelayError):
    pass


CHECKS = {
    "task_answer": {"required_fields"},
    "report_extraction": {"required_fields", "source_quotes_match"},
    "report_groups": {"required_fields", "report_ids_preserved"},
}


@dataclass(frozen=True)
class Task:
    id: str
    instruction: str
    preferred_tier: str
    input_refs: tuple[str, ...]
    depends_on: tuple[str, ...]
    result_type: str
    checks: tuple[str, ...]


@dataclass(frozen=True)
class Plan:
    tasks: tuple[Task, ...]
    final_instruction: str

    @classmethod
    def parse(cls, value: Any, sources: dict[str, str], max_tasks: int):
        if not isinstance(value, dict) or set(value) != {"version", "tasks", "final_instruction"}:
            raise ValidationError("Plan must contain version, tasks, and final_instruction")
        if type(value["version"]) is not int or value["version"] != 1:
            raise ValidationError("Unsupported plan version")
        rows = value["tasks"]
        if not isinstance(rows, list) or not 1 <= len(rows) <= max_tasks:
            raise ValidationError("Plan task count exceeds limits or is empty")
        if not isinstance(value["final_instruction"], str) or not 1 <= len(value["final_instruction"]) <= 2000:
            raise ValidationError("Invalid final instruction")
        tasks = []
        fields = set(Task.__dataclass_fields__)
        for row in rows:
            if not isinstance(row, dict) or set(row) != fields:
                raise ValidationError("Unexpected or missing task fields")
            for key in ("id", "instruction", "preferred_tier", "result_type"):
                if not isinstance(row[key], str) or not 1 <= len(row[key]) <= 2000:
                    raise ValidationError(f"Invalid task {key}")
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", row["id"]):
                raise ValidationError("Invalid task ID")
            if row["preferred_tier"] not in ("local", "cloud") or row["result_type"] not in CHECKS:
                raise ValidationError("Unknown task tier or result type")
            for key in ("input_refs", "depends_on", "checks"):
                values = row[key]
                if (not isinstance(values, list) or len(values) > 12
                        or any(not isinstance(v, str) for v in values)
                        or len(values) != len(set(values))):
                    raise ValidationError(f"Invalid task {key}")
            if set(row["checks"]) != CHECKS[row["result_type"]]:
                raise ValidationError("Required checks cannot be omitted or replaced")
            if not row["input_refs"]:
                raise ValidationError("Task needs inputs")
            tasks.append(Task(**{**row, **{key: tuple(row[key]) for key in ("input_refs", "depends_on", "checks")}}))
        ids = {task.id for task in tasks}
        if len(ids) != len(tasks):
            raise ValidationError("Duplicate task IDs")
        for task in tasks:
            if not set(task.depends_on) <= ids or task.id in task.depends_on:
                raise ValidationError("Unknown or self dependency")
            for ref in task.input_refs:
                prefix, _, key = ref.partition(".")
                if prefix == "sources" and key in sources:
                    continue
                if prefix == "results" and key in task.depends_on:
                    continue
                raise ValidationError("Unknown source or undeclared result dependency")
        resolved = set()
        while len(resolved) < len(tasks):
            ready = {t.id for t in tasks if t.id not in resolved and set(t.depends_on) <= resolved}
            if not ready:
                raise ValidationError("Cyclic task dependencies")
            resolved.update(ready)
        return cls(tuple(tasks), value["final_instruction"])


@dataclass(frozen=True)
class Reply:
    value: Any
    model: str
    # None is deliberately different from measured zero tokens.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class RunResult:
    run_id: str
    content: str
    results: dict
    metrics: dict
