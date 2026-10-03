import json

import pytest

from gaia_agent.store import RunStore
from gaia_agent.validation import TEST_COUNTS, validate_official_file
from scripts.merge_runs import merge_runs


def _levels():
    return {
        f"{level}-{number:03d}": level
        for level, count in TEST_COUNTS.items()
        for number in range(count)
    }


def _fingerprints(levels):
    return {task_id: f"input:{task_id}" for task_id in levels}


def _record(task_id, answer="first", signature="original", **changes):
    return {
        "task_id": task_id,
        "status": "completed",
        "answer": answer,
        "run_signature": signature,
        "input_fingerprint": f"input:{task_id}",
        "review_error": None,
        "adjudication_error": None,
        "warnings": [],
        **changes,
    }


def _full_run(path, levels):
    store = RunStore(path)
    for task_id in levels:
        store.put(task_id, _record(task_id))
    return store


def test_merge_uses_last_completed_answer_and_preserves_per_task_provenance(tmp_path):
    levels = _levels()
    first = _full_run(tmp_path / "first", levels)
    later = RunStore(tmp_path / "later")
    ids = list(levels)
    later.put(ids[0], _record(ids[0], "revised", "repair"))
    later.put(ids[1], _record(ids[1], "", "repair", status="error"))
    output = tmp_path / "gaia-test.jsonl"

    submission, manifest = merge_runs([first.root, later.root], levels, _fingerprints(levels), output)

    rows = validate_official_file(submission, levels)
    audit = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(audit) == 301
    assert rows[0] == {"task_id": ids[0], "model_answer": "revised"}
    assert rows[1] == {"task_id": ids[1], "model_answer": "first"}
    assert audit[0]["run_signature"] == "repair"
    assert audit[0]["run_directory"] == str(later.root.resolve())
    assert audit[1]["run_signature"] == "original"
    assert audit[1]["run_directory"] == str(first.root.resolve())
    assert len(audit[0]["record_sha256"]) == 64
    assert all(set(row) == {"task_id", "model_answer"} for row in rows)


@pytest.mark.parametrize("record,reason", [
    ({"status": "needs_adjudication", "adjudication_error": "failed"}, "no completed nonempty answer"),
    ({"status": "completed", "answer": ""}, "no completed nonempty answer"),
    ({"status": "completed", "answer": "  "}, "no completed nonempty answer"),
])
def test_merge_rejects_unanswered_task(tmp_path, record, reason):
    levels = _levels()
    task_id = next(iter(levels))
    store = _full_run(tmp_path / "run", levels)
    store.put(task_id, _record(task_id, **record))
    output = tmp_path / "submission.jsonl"

    with pytest.raises(ValueError, match=reason):
        merge_runs([store.root], levels, _fingerprints(levels), output)

    assert not output.exists()
    assert not output.with_suffix(".provenance.jsonl").exists()


@pytest.mark.parametrize("issue,reason", [
    ({"review_error": "timeout"}, "reviewer failed"),
    ({"adjudication_error": "timeout"}, "unresolved adjudication"),
    ({"warnings": ["attachment truncated"]}, "evidence issues"),
    ({"run_signature": ""}, "missing original run_signature"),
    ({"answer": "FINAL ANSWER: 42"}, "FINAL ANSWER"),
])
def test_merge_rejects_bad_selected_record_without_writing(tmp_path, issue, reason):
    levels = _levels()
    task_id = next(iter(levels))
    store = _full_run(tmp_path / "run", levels)
    store.put(task_id, _record(task_id, **issue))
    output = tmp_path / "submission.jsonl"

    with pytest.raises(ValueError, match=reason):
        merge_runs([store.root], levels, _fingerprints(levels), output)

    assert not output.exists()


def test_merge_rejects_missing_task_and_missing_directory(tmp_path):
    levels = _levels()
    output = tmp_path / "submission.jsonl"
    store = RunStore(tmp_path / "partial")
    task_id = next(iter(levels))
    store.put(task_id, _record(task_id))

    with pytest.raises(ValueError, match="no completed nonempty answer"):
        merge_runs([store.root], levels, _fingerprints(levels), output)
    with pytest.raises(FileNotFoundError, match="Run directory"):
        merge_runs([tmp_path / "absent"], levels, _fingerprints(levels), output)
    assert not output.exists()


def test_merge_rejects_stale_input_fingerprint(tmp_path):
    levels = _levels()
    task_id = next(iter(levels))
    store = _full_run(tmp_path / "run", levels)
    store.put(task_id, _record(task_id, input_fingerprint="old-input"))
    output = tmp_path / "submission.jsonl"
    with pytest.raises(ValueError, match="Input fingerprint differs"):
        merge_runs([store.root], levels, _fingerprints(levels), output)
    assert not output.exists()


def test_merge_rejects_degraded_evidence_even_without_warning(tmp_path):
    levels = _levels()
    task_id = next(iter(levels))
    store = _full_run(tmp_path / "run", levels)
    store.put(task_id, _record(task_id, final={"diagnostics": {"evidence_degraded": True}}))
    output = tmp_path / "submission.jsonl"
    with pytest.raises(ValueError, match="evidence issues"):
        merge_runs([store.root], levels, _fingerprints(levels), output)
    assert not output.exists()
