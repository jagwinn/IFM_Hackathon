"""Kinds of subtask results and the checks each must pass before the relay accepts it.

To add a kind: write a validate function, add a ResultType to RESULT_TYPES, and mention it in the planner
prompt (prompts.PLAN) if the cloud planner should be allowed to use it.
"""

from dataclasses import dataclass
from typing import Any, Callable

from .errors import ValidationError


@dataclass(frozen=True)
class ResultType:
    name: str
    checks: frozenset[str]  # the plan must list exactly these; they name what `validate` enforces
    worker_format: str  # tells the worker which JSON shape to return
    validate: Callable[[Any, dict[str, Any]], None]  # (value, inputs) -> None, raises ValidationError


def _task_answer(value, inputs):
    if set(value) != {"answer"} or not isinstance(value["answer"], str) or not 1 <= len(value["answer"].strip()) <= 16000:
        raise ValidationError("Task answer requires nonempty answer text of at most 16000 characters")


def _report_extraction(value, inputs):
    if set(value) != {"report_id", "quote"}:
        raise ValidationError("Extraction requires report_id and quote")
    report_id, quote = value["report_id"], value["quote"]
    if not isinstance(report_id, str) or not isinstance(quote, str) or not quote.strip():
        raise ValidationError("Invalid or empty report evidence")
    source = inputs.get(f"sources.{report_id}")
    if not isinstance(source, str) or quote not in source:
        raise ValidationError("Quote does not match the assigned source")
    source_refs = [ref for ref in inputs if ref.startswith("sources.")]
    if source_refs != [f"sources.{report_id}"]:
        raise ValidationError("Extraction must cover its single assigned source")


def _report_groups(value, inputs):
    if set(value) != {"groups"} or not isinstance(value["groups"], list) or not value["groups"]:
        raise ValidationError("Grouping requires nonempty groups")
    expected = {v["report_id"] for v in inputs.values() if isinstance(v, dict) and "report_id" in v}
    if not expected or len(expected) != len(inputs):
        raise ValidationError("Grouping needs one extraction per report")
    actual = []
    for group in value["groups"]:
        if not isinstance(group, dict) or set(group) != {"label", "report_ids"}:
            raise ValidationError("Invalid group fields")
        if not isinstance(group["label"], str) or not 1 <= len(group["label"]) <= 300:
            raise ValidationError("Invalid group label")
        ids = group["report_ids"]
        if not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids):
            raise ValidationError("Invalid group report IDs")
        actual.extend(ids)
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValidationError("Every report ID must occur exactly once")


RESULT_TYPES = {t.name: t for t in (
    ResultType("task_answer", frozenset({"required_fields"}),
               'Return exactly {"answer":"your subtask result"}.', _task_answer),
    ResultType("report_extraction", frozenset({"required_fields", "source_quotes_match"}),
               'Return exactly {"report_id":"source key","quote":"verbatim source text"}.', _report_extraction),
    ResultType("report_groups", frozenset({"required_fields", "report_ids_preserved"}),
               'Return exactly {"groups":[{"label":"category","report_ids":["report_1"]}]}; include each assigned report once.',
               _report_groups),
)}


def validate_result(result_type: str, value, inputs: dict) -> None:
    if result_type not in RESULT_TYPES:
        raise ValidationError("Unsupported result type")
    if not isinstance(value, dict):
        raise ValidationError("Worker must return an object")
    RESULT_TYPES[result_type].validate(value, inputs)


JUDGMENT_KEYS = ("confidence", "reason", "concern")


def split_judgment(value):
    """Separate a model's self-assessment from its answer: (answer without judgment keys, judgment or None).

    The judgment is shown to people but never used to accept a result; checks see only the answer.
    """
    if not isinstance(value, dict) or not any(k in value for k in JUDGMENT_KEYS):
        return value, None
    rest = {k: v for k, v in value.items() if k not in JUDGMENT_KEYS}
    level = value.get("confidence")
    note = value.get("reason") or value.get("concern") or ""
    judgment = {"confidence": level if level in ("low", "medium", "high") else None,
                "note": note.strip()[:300] if isinstance(note, str) else ""}
    return rest, judgment if judgment["confidence"] or judgment["note"] else None
