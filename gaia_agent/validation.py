"""课程与官方榜提交文件的离线校验。"""

from __future__ import annotations

import json
import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


TEST_COUNTS = {1: 93, 2: 159, 3: 49}
WAIVABLE_ISSUES = {"warnings", "evidence_degraded", "omitted_sources"}


def _answer(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("model_answer/submitted_answer must be a string")
    value = value.strip()
    if not value or "\n" in value or "\r" in value or re.match(r"(?i)^FINAL\s+ANSWER\b", value):
        raise ValueError("answer must be nonempty, one line, and contain no FINAL ANSWER prefix")
    if len(value) > 500:
        raise ValueError("answer is unexpectedly long")
    return value


def validate_official_records(records: Iterable[dict[str, Any]], expected: dict[str, int] | None = None) -> list[dict[str, str]]:
    rows = list(records)
    seen: set[str] = set()
    result = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != {"task_id", "model_answer"}:
            raise ValueError(f"line {index}: exactly task_id and model_answer are required")
        task_id = row["task_id"]
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"line {index}: task_id must be a nonempty string")
        if task_id in seen:
            raise ValueError(f"duplicate task_id: {task_id}")
        seen.add(task_id)
        result.append({"task_id": task_id, "model_answer": _answer(row["model_answer"])})
    if expected is not None:
        if len(expected) != 301 or Counter(expected.values()) != Counter(TEST_COUNTS):
            raise ValueError("official test manifest must have 301 tasks at levels 93/159/49")
        if seen != set(expected):
            missing = sorted(set(expected) - seen)
            extra = sorted(seen - set(expected))
            raise ValueError(f"task ID coverage mismatch; missing={missing[:3]}, extra={extra[:3]}")
    return result


def validate_official_file(path: str | Path, task_levels: dict[str, int]) -> list[dict[str, str]]:
    records = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
    return validate_official_records(records, expected=task_levels)


def write_official(path: str | Path, records: Iterable[dict[str, Any]], task_levels: dict[str, int]) -> None:
    rows = validate_official_records(records, expected=task_levels)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def evidence_issues(record: dict[str, Any]) -> set[str]:
    """Return evidence limitations that require individual human review."""
    issues = set()
    if record.get("warnings"):
        issues.add("warnings")
    for result in (record.get("primary"), record.get("final")):
        if not isinstance(result, dict):
            continue
        diagnostics = result.get("diagnostics") or {}
        if diagnostics.get("evidence_degraded") or diagnostics.get("final_input_compacted"):
            issues.add("evidence_degraded")
        if diagnostics.get("omitted_sources"):
            issues.add("omitted_sources")
    return issues


def evidence_fingerprint(record: dict[str, Any]) -> str:
    details = {
        "warnings": record.get("warnings") or [],
        "primary": (record.get("primary") or {}).get("diagnostics") or {},
        "final": (record.get("final") or {}).get("diagnostics") or {},
    }
    payload = json.dumps(details, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_review_waivers(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Review waivers must be a JSON object keyed by task_id")
    return value


def check_export_evidence(records: list[dict[str, Any]], waivers: dict[str, Any]) -> None:
    used = set()
    for record in records:
        task_id = record["task_id"]
        if record.get("review_error"):
            raise RuntimeError(f"{task_id}: reviewer failed; rerun before export")
        if record.get("adjudication_error"):
            raise RuntimeError(f"{task_id}: unresolved adjudication; rerun before export")
        issues = evidence_issues(record)
        if not issues:
            continue
        waiver = waivers.get(task_id)
        evidence_hash = evidence_fingerprint(record)
        if not isinstance(waiver, dict):
            raise RuntimeError(
                f"{task_id}: evidence issues {sorted(issues)}; evidence_fingerprint={evidence_hash}; "
                "review and provide a per-task --review-waivers file"
            )
        if (set(waiver) != {"run_signature", "input_fingerprint", "evidence_fingerprint", "answer", "allow", "reason"}
                or waiver.get("run_signature") != record.get("run_signature")
                or waiver.get("input_fingerprint") != record.get("input_fingerprint")
                or waiver.get("evidence_fingerprint") != evidence_hash
                or waiver.get("answer") != record.get("answer")
                or not isinstance(waiver.get("allow"), list)
                or not all(isinstance(item, str) for item in waiver["allow"])
                or set(waiver["allow"]) != issues
                or not issues <= WAIVABLE_ISSUES
                or not isinstance(waiver.get("reason"), str)
                or len(waiver["reason"].strip()) < 10):
            raise RuntimeError(f"{task_id}: review waiver does not match the current answer, evidence and issues")
        used.add(task_id)
    if set(waivers) != used:
        raise RuntimeError(f"Unused or stale review waivers: {sorted(set(waivers) - used)[:3]}")


def validate_stored_record(
    record: dict[str, Any] | None, task_id: str, expected_fingerprint: str,
    expected_signature: str | None = None,
) -> str:
    """Apply the same provenance and answer checks to single and merged exports."""
    if record is None:
        raise RuntimeError(f"Task {task_id} has never run; run it before export")
    if record.get("task_id") != task_id:
        raise RuntimeError(f"Stored task ID differs for {task_id}; rerun before export")
    signature = record.get("run_signature")
    if not isinstance(signature, str) or not signature:
        raise RuntimeError(f"{task_id}: missing original run_signature")
    if expected_signature is not None and signature != expected_signature:
        raise RuntimeError(f"Run settings differ for {task_id}; use the original config or a new --runs directory")
    if record.get("input_fingerprint") != expected_fingerprint:
        raise RuntimeError(f"Input fingerprint differs for {task_id}; rerun it before export")
    if record.get("status") != "completed":
        raise RuntimeError(f"Task {task_id} has status {record.get('status')!r}; retry it before export")
    try:
        return _answer(record.get("answer"))
    except ValueError as exc:
        raise RuntimeError(f"{task_id}: {exc}") from exc


def course_payload(
    username: str,
    agent_code: str,
    records: Iterable[dict[str, Any]],
    *,
    expected_ids: set[str] | None = None,
) -> dict[str, Any]:
    answers = []
    for row in records:
        if not isinstance(row, dict) or "task_id" not in row or "answer" not in row:
            raise ValueError("course record requires task_id and answer")
        answer = _answer(row["answer"])
        answers.append({"task_id": str(row["task_id"]), "submitted_answer": answer})
    if len(answers) != 20:
        raise ValueError(f"course payload requires exactly 20 answers, got {len(answers)}")
    if len({answer["task_id"] for answer in answers}) != 20:
        raise ValueError("course payload contains duplicate task IDs")
    if expected_ids is not None and {answer["task_id"] for answer in answers} != expected_ids:
        raise ValueError("course payload task IDs do not match the current course questions")
    if not username or not agent_code.startswith("https://huggingface.co/spaces/"):
        raise ValueError("username and public Hugging Face Space agent_code are required")
    code_url = agent_code.rstrip("/")
    if not code_url.endswith("/tree/main"):
        code_url += "/tree/main"
    return {"username": username, "agent_code": code_url, "answers": answers}
