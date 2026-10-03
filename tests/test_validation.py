import json
from collections import Counter

import pytest

from gaia_agent.validation import (
    TEST_COUNTS,
    course_payload,
    validate_official_file,
    validate_official_records,
    write_official,
)


def make_official():
    levels = {}
    for level, count in TEST_COUNTS.items():
        for number in range(count):
            levels[f"{level}-{number:03d}"] = level
    rows = [{"task_id": task_id, "model_answer": "42"} for task_id in levels]
    return rows, levels


def test_official_round_trip_and_level_contract(tmp_path):
    rows, levels = make_official()
    target = tmp_path / "submission.jsonl"
    write_official(target, rows, levels)
    checked = validate_official_file(target, levels)
    assert len(checked) == 301
    assert Counter(levels.values()) == Counter(TEST_COUNTS)
    assert all(set(row) == {"task_id", "model_answer"} for row in checked)


@pytest.mark.parametrize("change", ["duplicate", "missing", "extra"])
def test_official_rejects_invalid_export(change):
    rows, levels = make_official()
    if change == "duplicate":
        rows[-1]["task_id"] = rows[0]["task_id"]
    elif change == "missing":
        rows.pop()
    else:
        rows.append({"task_id": "extra", "model_answer": "42"})
    with pytest.raises(ValueError):
        validate_official_records(rows, levels)


def test_official_rejects_malformed_jsonl(tmp_path):
    _, levels = make_official()
    target = tmp_path / "bad.jsonl"
    target.write_text('{"task_id":', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        validate_official_file(target, levels)


def test_official_rejects_extra_fields():
    rows, levels = make_official()
    rows[0]["reasoning_trace"] = "private details"
    with pytest.raises(ValueError, match="exactly task_id and model_answer"):
        validate_official_records(rows, levels)


def test_official_rejects_blank_answer():
    rows, levels = make_official()
    rows[0]["model_answer"] = ""
    with pytest.raises(ValueError, match="nonempty"):
        validate_official_records(rows, levels)


@pytest.mark.parametrize("answer", ["final answer: 42", "first\rsecond"])
def test_official_rejects_prefix_or_internal_carriage_return(answer):
    rows, levels = make_official()
    rows[0]["model_answer"] = answer
    with pytest.raises(ValueError, match="one line|FINAL ANSWER"):
        validate_official_records(rows, levels)


def test_official_allows_final_answer_phrase_after_start():
    rows, levels = make_official()
    rows[0]["model_answer"] = "A book called Final Answer"
    assert validate_official_records(rows, levels)[0]["model_answer"] == "A book called Final Answer"


def test_course_payload_exact_20_and_space_url():
    rows = [{"task_id": str(number), "answer": str(number)} for number in range(20)]
    payload = course_payload("student", "https://huggingface.co/spaces/student/agent", rows)
    assert payload["agent_code"].endswith("/tree/main")
    assert len(payload["answers"]) == 20
    assert set(payload["answers"][0]) == {"task_id", "submitted_answer"}
    with pytest.raises(ValueError, match="duplicate"):
        course_payload("student", "https://huggingface.co/spaces/student/agent", rows[:-1] + rows[:1])


def test_course_rejects_final_answer_prefix():
    rows = [{"task_id": str(number), "answer": "yes"} for number in range(20)]
    rows[0]["answer"] = "FINAL ANSWER: yes"
    with pytest.raises(ValueError):
        course_payload("student", "https://huggingface.co/spaces/student/agent", rows)


def test_course_payload_checks_current_task_ids_when_supplied():
    rows = [{"task_id": str(number), "answer": str(number)} for number in range(20)]
    with pytest.raises(ValueError, match="task IDs"):
        course_payload("student", "https://huggingface.co/spaces/student/agent", rows, expected_ids={"other"})
