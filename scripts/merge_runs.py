"""Merge completed GAIA test runs without changing their original provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from gaia_agent.config import load_config
from gaia_agent.cli import _prepare_inputs
from gaia_agent.sources import gaia_tasks
from gaia_agent.store import RunStore
from gaia_agent.validation import (
    check_export_evidence, evidence_fingerprint, load_review_waivers,
    validate_official_records, validate_stored_record, write_official,
)


def merge_runs(
    run_directories: list[str | Path],
    task_levels: dict[str, int],
    task_fingerprints: dict[str, str],
    output: str | Path,
    provenance_output: str | Path | None = None,
    review_waivers: dict | None = None,
) -> tuple[Path, Path]:
    """Select the newest usable record in directory order and export a full test set."""
    if not run_directories:
        raise ValueError("At least one run directory is required")
    if set(task_levels) != set(task_fingerprints):
        raise ValueError("Current input fingerprints must cover the complete test manifest")
    roots = [Path(directory).resolve() for directory in run_directories]
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {root}")
    target = Path(output).resolve()
    provenance_target = (
        Path(provenance_output).resolve()
        if provenance_output is not None
        else target.with_suffix(".provenance.jsonl")
    )
    if target == provenance_target:
        raise ValueError("Submission and provenance paths must differ")

    stores = [RunStore(root) for root in roots]
    answers: list[dict[str, str]] = []
    provenance: list[dict[str, str]] = []
    selected_records: list[dict] = []
    failures: list[str] = []
    for task_id in task_levels:
        selected = None
        for root, store in zip(roots, stores):
            record = store.get(task_id)
            if record is None:
                continue
            if record.get("task_id") != task_id:
                failures.append(f"{task_id}: record task_id mismatch in {root}")
                continue
            answer = record.get("answer")
            if record.get("status") == "completed" and isinstance(answer, str) and answer.strip():
                selected = (root, record)
        if selected is None:
            failures.append(f"{task_id}: no completed nonempty answer")
            continue

        root, record = selected
        try:
            answer = validate_stored_record(record, task_id, task_fingerprints[task_id])
        except RuntimeError as exc:
            failures.append(str(exc))
            continue
        signature = record["run_signature"]
        record_file = root / f"{task_id}.json"
        answers.append({"task_id": task_id, "model_answer": answer})
        selected_records.append(record)
        provenance.append({
            "task_id": task_id,
            "run_directory": str(root),
            "record_file": str(record_file),
            "run_signature": signature,
            "input_fingerprint": record["input_fingerprint"],
            "evidence_fingerprint": evidence_fingerprint(record),
            "record_sha256": hashlib.sha256(record_file.read_bytes()).hexdigest(),
        })

    if failures:
        preview = "; ".join(failures[:10])
        raise ValueError(f"Merge rejected {len(failures)} issue(s): {preview}")
    try:
        check_export_evidence(selected_records, review_waivers or {})
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    checked = validate_official_records(answers, expected=task_levels)
    if any(not row["model_answer"] for row in checked):
        raise ValueError("Merge contains an empty answer")

    # Prepare both outputs before replacing either destination.
    target.parent.mkdir(parents=True, exist_ok=True)
    provenance_target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".jsonl.tmp", delete=False) as handle:
        temporary_answer = Path(handle.name)
    with tempfile.NamedTemporaryFile(dir=provenance_target.parent, suffix=".jsonl.tmp", delete=False) as handle:
        temporary_provenance = Path(handle.name)
    try:
        write_official(temporary_answer, checked, task_levels)
        with temporary_provenance.open("w", encoding="utf-8", newline="\n") as handle:
            for row in provenance:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temporary_provenance, provenance_target)
        os.replace(temporary_answer, target)
    finally:
        temporary_answer.unlink(missing_ok=True)
        temporary_provenance.unlink(missing_ok=True)
    return target, provenance_target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", action="append", required=True, help="RunStore directory; repeat in oldest-to-newest order")
    parser.add_argument("--output", required=True, help="Official submission JSONL path")
    parser.add_argument("--provenance", help="Private provenance JSONL path; defaults beside --output")
    parser.add_argument("--review-waivers", help="Reviewed per-task evidence exceptions, bound to selected records")
    parser.add_argument("--config", default="config.toml")
    args = parser.parse_args()
    config = load_config(args.config)
    tasks = gaia_tasks("test", config.hf_token or None, download=False)
    fingerprints = _prepare_inputs(tasks, "test", config)
    submission, provenance = merge_runs(
        args.runs,
        {task.task_id: task.level for task in tasks},
        fingerprints,
        args.output,
        args.provenance,
        load_review_waivers(args.review_waivers),
    )
    print(f"Wrote {submission} and {provenance} ({len(tasks)} answers)")


if __name__ == "__main__":
    main()
