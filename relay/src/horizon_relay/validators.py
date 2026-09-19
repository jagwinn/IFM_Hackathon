from .schemas import Task, ValidationError


def validate_result(task: Task, value, inputs: dict):
    if not isinstance(value, dict):
        raise ValidationError("Worker must return an object")
    if task.result_type == "task_answer":
        if set(value) != {"answer"} or not isinstance(value["answer"], str) or not 1 <= len(value["answer"].strip()) <= 16000:
            raise ValidationError("Task answer requires nonempty answer text of at most 16000 characters")
    elif task.result_type == "report_extraction":
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
    elif task.result_type == "report_groups":
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
    else:
        raise ValidationError("Unsupported result type")
    return value
