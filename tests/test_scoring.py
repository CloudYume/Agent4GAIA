from gaia_agent.scoring import question_scorer, score_records


def test_official_number_and_ordered_list_matching():
    assert question_scorer("$1,000", "1000")
    assert question_scorer("Red; Blue", "red,blue")
    assert not question_scorer("Blue, Red", "red,blue")
    assert not question_scorer("3.1", "3.10, 4")


def test_score_records_reports_attachment_breakdown():
    rows = [
        {"task_id": "a", "Level": 1, "Final answer": "42", "file_name": ""},
        {"task_id": "b", "Level": 2, "Final answer": "yes", "file_name": "file.pdf"},
    ]
    records = {"a": {"status": "completed", "answer": "42"}, "b": {"status": "error"}}
    report = score_records(rows, records)
    assert report["score_percent"] == 50.0
    assert report["by_attachment"]["with_file"]["correct"] == 0


def test_course_exact_estimate_does_not_use_official_normalization():
    rows = [{"task_id": "a", "Level": 1, "Final answer": "New York", "file_name": ""}]
    records = {"a": {"status": "completed", "answer": "new york"}}
    assert score_records(rows, records)["correct"] == 1
    course = score_records(rows, records, mode="course_exact")
    assert course["correct"] == 0
    assert course["scoring_mode"] == "course_exact"


def test_score_records_distinguishes_failed_from_unrun():
    rows = [
        {"task_id": "a", "Level": 1, "Final answer": "yes", "file_name": ""},
        {"task_id": "b", "Level": 1, "Final answer": "no", "file_name": ""},
    ]
    report = score_records(rows, {"a": {"status": "error"}})
    assert report["run_status"] == {"error": 1, "not_run": 1}
