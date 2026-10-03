import argparse
import json

import pytest

from gaia_agent import cli
from gaia_agent.config import load_config
from gaia_agent.runmeta import input_fingerprint
from gaia_agent.sources import Task
from gaia_agent.store import RunStore


def test_doctor_reports_request_budget(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda path: load_config("config.example.toml"))
    cli._doctor(argparse.Namespace(config="unused"))
    report = json.loads(capsys.readouterr().out)
    assert report["search_context_size"] == "medium"
    assert report["max_tool_calls"] == 12
    assert report["max_output_tokens"] == 8000


def test_export_requires_all_tasks_completed(tmp_path):
    store = RunStore(tmp_path)
    tasks = [Task("a", "First?", 1), Task("b", "Second?", 1)]
    with pytest.raises(RuntimeError, match="never run"):
        cli._records(tmp_path, tasks, "sig")
    store.put("a", {"task_id": "a", "status": "error", "run_signature": "sig"})
    store.put("b", {"task_id": "b", "status": "completed", "answer": "yes", "run_signature": "sig"})
    with pytest.raises(RuntimeError, match="status 'error'"):
        cli._records(tmp_path, tasks, "sig")


def test_official_export_requires_exact_per_task_evidence_waiver(monkeypatch, tmp_path, capsys):
    tasks = [Task("a", "First?", 1), Task("b", "Second?", 2)]
    store = RunStore(tmp_path)
    record = {
        "task_id": "a", "status": "completed", "answer": "yes", "warnings": ["sampled pages"],
        "run_signature": "sig", "input_fingerprint": input_fingerprint(tasks[0]),
    }
    store.put("a", record)
    store.put("b", {
        "task_id": "b", "status": "completed", "answer": "no", "run_signature": "sig",
        "input_fingerprint": input_fingerprint(tasks[1]),
    })
    monkeypatch.setattr(cli, "load_config", lambda path: object())
    monkeypatch.setattr(cli, "run_signature", lambda config: "sig")
    monkeypatch.setattr(cli, "_tasks", lambda kind, config, download: tasks)
    store.ensure_snapshot(cli._snapshot_data(
        "test", tasks, {task.task_id: input_fingerprint(task) for task in tasks}, object(), "sig"
    ))
    written = []
    monkeypatch.setattr(cli, "write_official", lambda path, rows, levels: written.extend(rows))
    args = argparse.Namespace(
        config="unused", split="test", runs=tmp_path, output=str(tmp_path / "out.jsonl"),
        fill_failures=False, accept_partial_evidence=False, review_waivers=None,
    )
    with pytest.raises(RuntimeError, match="evidence issues"):
        cli._export(args)
    assert not written
    waiver = {"a": {
        "run_signature": "sig", "input_fingerprint": record["input_fingerprint"],
        "evidence_fingerprint": cli._evidence_fingerprint(record),
        "answer": "yes", "allow": ["warnings"],
        "reason": "Inspected every sampled page against the original PDF.",
    }}
    waiver_path = tmp_path / "waivers.json"
    waiver_path.write_text(json.dumps(waiver), encoding="utf-8")
    args.review_waivers = str(waiver_path)
    cli._export(args)
    assert [row["model_answer"] for row in written] == ["yes", "no"]
    assert "warnings=1, review_errors=0, unresolved=0, blank_answers=0" in capsys.readouterr().out

    waiver["a"]["answer"] = "changed"
    waiver_path.write_text(json.dumps(waiver), encoding="utf-8")
    with pytest.raises(RuntimeError, match="waiver does not match"):
        cli._export(args)


def test_official_export_never_includes_unadjudicated_task(monkeypatch, tmp_path):
    task = Task("a", "Question?", 1)
    RunStore(tmp_path).put("a", {
        "task_id": "a", "status": "needs_adjudication", "answer": "draft",
        "adjudication_error": "timeout", "run_signature": "sig",
        "input_fingerprint": input_fingerprint(task),
    })
    monkeypatch.setattr(cli, "load_config", lambda path: object())
    monkeypatch.setattr(cli, "run_signature", lambda config: "sig")
    monkeypatch.setattr(cli, "_tasks", lambda kind, config, download: [task])
    RunStore(tmp_path).ensure_snapshot(cli._snapshot_data(
        "test", [task], {task.task_id: input_fingerprint(task)}, object(), "sig"
    ))
    args = argparse.Namespace(
        config="unused", split="test", runs=tmp_path, output=str(tmp_path / "out.jsonl"),
        fill_failures=False, accept_partial_evidence=False, review_waivers=None,
    )
    with pytest.raises(RuntimeError, match="needs_adjudication"):
        cli._export(args)
    assert not (tmp_path / "out.jsonl").exists()


def test_run_bounds_in_flight_tasks_and_retries_review_errors_explicitly(monkeypatch, tmp_path):
    tasks = [Task(str(index), f"Question {index}?", 1) for index in range(5)]
    store = RunStore(tmp_path)
    store.put("0", {"status": "completed", "answer": "old", "review_error": "timeout", "run_signature": "sig", "input_fingerprint": input_fingerprint(tasks[0])})
    store.put("1", {"status": "completed", "answer": "old", "run_signature": "sig", "input_fingerprint": input_fingerprint(tasks[1])})
    monkeypatch.setattr(cli, "load_config", lambda path: type("Config", (), {"workers": 2, "hf_token": ""})())
    monkeypatch.setattr(cli, "run_signature", lambda config: "sig")
    monkeypatch.setattr(cli, "_tasks", lambda kind, config, download: tasks)
    solved = []

    class Agent:
        def __init__(self, config):
            pass

        def solve(self, task, *, previous=None, checkpoint=None):
            solved.append(task.task_id)
            return {"status": "completed", "answer": task.task_id}

    monkeypatch.setattr(cli, "GaiaAgent", Agent)
    actual_wait = cli.wait

    def bounded_wait(active, *, return_when):
        assert len(active) <= 2
        return actual_wait(active, return_when=return_when)

    monkeypatch.setattr(cli, "wait", bounded_wait)
    args = argparse.Namespace(
        config="unused", split="course", no_download=True, runs=tmp_path,
        force=False, retry_review_errors=True, task_id=None, limit=None,
    )
    cli._run(args)
    assert set(solved) == {"0", "2", "3", "4"}
    assert store.get("0")["answer"] == "0"
    assert store.get("1")["answer"] == "old"


def test_interrupted_run_checkpoints_active_tasks(monkeypatch, tmp_path):
    tasks = [Task(str(index), f"Question {index}?", 1) for index in range(4)]
    monkeypatch.setattr(cli, "load_config", lambda path: type("Config", (), {"workers": 2, "hf_token": ""})())
    monkeypatch.setattr(cli, "run_signature", lambda config: "sig")
    monkeypatch.setattr(cli, "_tasks", lambda kind, config, download: tasks)

    class Agent:
        def __init__(self, config):
            pass

        def solve(self, task, *, previous=None, checkpoint=None):
            return {"status": "completed", "answer": task.task_id}

    monkeypatch.setattr(cli, "GaiaAgent", Agent)

    def interrupt(active, *, return_when):
        assert len(active) == 2
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "wait", interrupt)
    args = argparse.Namespace(
        config="unused", split="course", no_download=True, runs=tmp_path,
        force=False, retry_review_errors=False, task_id=None, limit=None,
    )
    with pytest.raises(KeyboardInterrupt):
        cli._run(args)
    store = RunStore(tmp_path)
    assert [store.get(str(index))["answer"] for index in range(2)] == ["0", "1"]
    assert all(store.get(str(index)) is None for index in range(2, 4))


def test_run_stops_after_repeated_systemic_errors(monkeypatch, tmp_path):
    tasks = [Task(str(index), f"Question {index}?", 1) for index in range(20)]
    monkeypatch.setattr(cli, "load_config", lambda path: type("Config", (), {"workers": 2, "hf_token": ""})())
    monkeypatch.setattr(cli, "run_signature", lambda config: "sig")
    monkeypatch.setattr(cli, "_tasks", lambda kind, config, download: tasks)

    class Agent:
        def __init__(self, config):
            pass

        def solve(self, task, *, previous=None, checkpoint=None):
            raise ValueError("same model protocol failure")

    monkeypatch.setattr(cli, "GaiaAgent", Agent)
    args = argparse.Namespace(
        config="unused", split="course", no_download=True, runs=tmp_path,
        force=False, retry_review_errors=False, task_id=None, limit=None,
    )
    with pytest.raises(RuntimeError, match="circuit breaker"):
        cli._run(args)
    saved = list(tmp_path.glob("*.json"))
    assert 3 <= len(saved) <= 5
    assert all(json.loads(path.read_text(encoding="utf-8"))["status"] == "error" for path in saved)


def test_course_run_checks_attachments_even_for_completed_tasks(monkeypatch, tmp_path):
    tasks = [Task(str(index), f"Question {index}?", 1, file_name=f"{index}.png") for index in range(3)]
    for task in tasks:
        path = tmp_path / task.file_name
        path.write_bytes(task.task_id.encode())
        task.attachment_path = str(path)
    store = RunStore(tmp_path)
    store.put("0", {"status": "completed", "answer": "old", "run_signature": "sig", "input_fingerprint": input_fingerprint(tasks[0])})
    monkeypatch.setattr(cli, "load_config", lambda path: type("Config", (), {"workers": 1, "hf_token": ""})())
    monkeypatch.setattr(cli, "run_signature", lambda config: "sig")
    monkeypatch.setattr(cli, "_tasks", lambda kind, config, download: tasks)
    requested = []

    def download(selected, token):
        requested.extend(task.task_id for task in selected)

    monkeypatch.setattr(cli, "download_course_attachments", download)

    class Agent:
        def __init__(self, config):
            pass

        def solve(self, task, *, previous=None, checkpoint=None):
            return {"status": "completed", "answer": task.task_id}

    monkeypatch.setattr(cli, "GaiaAgent", Agent)
    args = argparse.Namespace(
        config="unused", split="course", no_download=False, runs=tmp_path,
        force=False, retry_review_errors=False, task_id=None, limit=1,
    )
    cli._run(args)
    assert requested == ["0", "1", "2"]


@pytest.mark.parametrize("change", ["question", "attachment"])
def test_run_does_not_reuse_primary_after_input_change(monkeypatch, tmp_path, change):
    attachment = tmp_path / "evidence.txt"
    attachment.write_text("old bytes", encoding="utf-8")
    task = Task("a", "Original question?", 1, file_name="evidence.txt", attachment_path=str(attachment))
    old_fingerprint = input_fingerprint(task)
    store = RunStore(tmp_path / "records")
    store.put("a", {
        "task_id": "a", "status": "primary_completed", "answer": "old",
        "primary": {"answer": "old"}, "run_signature": "sig",
        "input_fingerprint": old_fingerprint,
    })
    if change == "question":
        task.question = "Changed question?"
    else:
        attachment.write_text("new bytes", encoding="utf-8")
    monkeypatch.setattr(cli, "load_config", lambda path: type("Config", (), {"workers": 1, "hf_token": ""})())
    monkeypatch.setattr(cli, "run_signature", lambda config: "sig")
    monkeypatch.setattr(cli, "_tasks", lambda kind, config, download: [task])
    monkeypatch.setattr(cli, "download_course_attachments", lambda tasks, token: None)
    observed = []

    class Agent:
        def __init__(self, config):
            pass

        def solve(self, task, *, previous=None, checkpoint=None):
            observed.append(previous)
            checkpoint({"status": "primary_completed", "answer": "new", "primary": {"answer": "new"}})
            return {"status": "completed", "answer": "new"}

    monkeypatch.setattr(cli, "GaiaAgent", Agent)
    args = argparse.Namespace(
        config="unused", split="course", no_download=False, runs=store.root,
        force=False, retry_review_errors=False, task_id=None, limit=None,
    )
    cli._run(args)
    assert observed == [None]
    saved = store.get("a")
    assert saved["answer"] == "new"
    assert saved["input_fingerprint"] == input_fingerprint(task)
    assert saved["input_fingerprint"] != old_fingerprint


def test_force_failure_does_not_retain_old_primary(monkeypatch, tmp_path):
    task = Task("a", "Question?", 1)
    store = RunStore(tmp_path)
    store.put("a", {
        "task_id": "a", "status": "completed", "answer": "old",
        "primary": {"answer": "old"}, "run_signature": "sig",
        "input_fingerprint": input_fingerprint(task),
    })
    monkeypatch.setattr(cli, "load_config", lambda path: type("Config", (), {"workers": 1, "hf_token": ""})())
    monkeypatch.setattr(cli, "run_signature", lambda config: "sig")
    monkeypatch.setattr(cli, "_tasks", lambda kind, config, download: [task])

    class Agent:
        def __init__(self, config):
            pass

        def solve(self, task, *, previous=None, checkpoint=None):
            assert previous is None
            raise RuntimeError("new attempt failed before checkpoint")

    monkeypatch.setattr(cli, "GaiaAgent", Agent)
    args = argparse.Namespace(
        config="unused", split="course", no_download=True, runs=store.root,
        force=True, retry_review_errors=False, task_id=None, limit=None,
    )
    cli._run(args)
    saved = store.get("a")
    assert saved["status"] == "error"
    assert "primary" not in saved
    assert saved["input_fingerprint"] == input_fingerprint(task)


def test_stage_checkpoint_survives_failure_and_is_reused(monkeypatch, tmp_path):
    task = Task("a", "Question?", 1)
    store = RunStore(tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda path: type("Config", (), {"workers": 1, "hf_token": ""})())
    monkeypatch.setattr(cli, "run_signature", lambda config: "sig")
    monkeypatch.setattr(cli, "_tasks", lambda kind, config, download: [task])
    previous_records = []

    class Agent:
        def __init__(self, config):
            pass

        def solve(self, task, *, previous=None, checkpoint=None):
            previous_records.append(previous)
            if previous is None:
                checkpoint({"status": "primary_completed", "answer": "42", "primary": {"answer": "42"}})
                raise RuntimeError("review interrupted")
            return {"status": "completed", "answer": previous["primary"]["answer"]}

    monkeypatch.setattr(cli, "GaiaAgent", Agent)
    args = argparse.Namespace(
        config="unused", split="course", no_download=True, runs=store.root,
        force=False, retry_review_errors=False, task_id=None, limit=None,
    )
    cli._run(args)
    failed = store.get("a")
    assert failed["status"] == "error"
    assert failed["primary"]["answer"] == "42"
    assert failed["input_fingerprint"] == input_fingerprint(task)
    cli._run(args)
    assert previous_records[0] is None
    assert previous_records[1]["primary"]["answer"] == "42"
    assert store.get("a")["status"] == "completed"


def test_export_rejects_degraded_evidence_without_matching_waiver():
    record = {
        "task_id": "a", "answer": "42", "run_signature": "sig", "input_fingerprint": "input",
        "primary": {"diagnostics": {"final_trigger": "tool_call_limit"}},
        "final": {"diagnostics": {"evidence_degraded": True, "omitted_sources": ["https://example.com/"]}},
    }
    assert cli._evidence_issues({**record, "final": {"diagnostics": {"final_trigger": "tool_call_limit"}}}) == set()
    with pytest.raises(RuntimeError, match="evidence issues"):
        cli._check_export_evidence([record], {})
    waiver = {"a": {
        "run_signature": "sig", "input_fingerprint": "input", "answer": "42",
        "evidence_fingerprint": cli._evidence_fingerprint(record),
        "allow": ["evidence_degraded", "omitted_sources"],
        "reason": "Inspected the complete source text and confirmed the answer.",
    }}
    cli._check_export_evidence([record], waiver)
    waiver["a"]["evidence_fingerprint"] = "stale"
    with pytest.raises(RuntimeError, match="waiver does not match"):
        cli._check_export_evidence([record], waiver)


def test_export_rejects_global_partial_evidence_and_blank_fill(tmp_path):
    args = argparse.Namespace(
        split="test", fill_failures=True, accept_partial_evidence=False,
        review_waivers=None,
    )
    with pytest.raises(ValueError, match="strict official submission"):
        cli._export(args)
    args.fill_failures = False
    args.accept_partial_evidence = True
    with pytest.raises(ValueError, match="per-task"):
        cli._export(args)


def test_pilot_manifest_is_deterministic_and_covers_levels_and_youtube():
    counts = {1: 93, 2: 159, 3: 49}
    tasks = [
        Task(f"{level}-{index:03d}", f"Question {level}/{index}?", level)
        for level, count in counts.items() for index in range(count)
    ]
    tasks[0].question += " https://www.youtube.com/watch?v=example"
    tasks[1].file_name = "audio.mp3"
    tasks[94].file_name = "image.png"
    snapshot = {
        "split": "test",
        "run_signature": "sig", "dataset_fingerprint": "dataset",
        "input_manifest_fingerprint": "inputs",
    }
    first = cli._pilot_manifest(tasks, snapshot)
    assert first == cli._pilot_manifest(tasks, snapshot)
    assert len(first["task_ids"]) == len(set(first["task_ids"])) == 20
    assert first["level_counts"] == {"1": 6, "2": 10, "3": 4}
    assert first["evidence_counts"]["youtube"] == 1
    assert first["evidence_counts"]["audio"] == 1


def test_manifest_rejects_stale_run_and_unknown_ids(tmp_path):
    tasks = [Task("a", "First?", 1), Task("b", "Second?", 2)]
    snapshot = {
        "split": "test", "run_signature": "sig", "dataset_fingerprint": "dataset",
        "input_manifest_fingerprint": "inputs",
    }
    manifest = {**snapshot, "schema_version": 1, "task_ids": ["a"]}
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert [task.task_id for task in cli._manifest_tasks(str(path), tasks, snapshot)] == ["a"]
    manifest["run_signature"] = "other"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs"):
        cli._manifest_tasks(str(path), tasks, snapshot)
    manifest["run_signature"] = "sig"
    manifest["task_ids"] = ["unknown"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        cli._manifest_tasks(str(path), tasks, snapshot)


def test_status_report_counts_unfinished_and_evidence_issues(tmp_path):
    tasks = [Task("a", "First?", 1), Task("b", "Second?", 2), Task("c", "Third?", 3)]
    fingerprints = {task.task_id: input_fingerprint(task) for task in tasks}
    store = RunStore(tmp_path)
    store.put("a", {
        "task_id": "a", "status": "completed", "answer": "yes",
        "run_signature": "sig", "input_fingerprint": fingerprints["a"],
        "warnings": ["sampled evidence"],
    })
    store.put("b", {
        "task_id": "b", "status": "error", "run_signature": "sig",
        "input_fingerprint": fingerprints["b"],
    })
    report = cli._result_report(tasks, store, {
        "split": "test", "run_signature": "sig", "dataset_fingerprint": "dataset",
        "input_manifest_fingerprint": "inputs", "input_fingerprints": fingerprints,
    })
    assert report["status_counts"] == {"completed": 1, "error": 1, "pending": 1}
    assert report["evidence_issue_counts"] == {"warnings": 1}
    assert report["unfinished_task_ids"] == ["b", "c"]
    assert report["strict_ready_without_waivers"] is False


def test_run_snapshot_rejects_changed_inputs(tmp_path):
    store = RunStore(tmp_path)
    expected = {"schema_version": 1, "run_signature": "sig", "input_manifest_fingerprint": "first"}
    store.ensure_snapshot(expected)
    assert store.snapshot()["run_signature"] == "sig"
    with pytest.raises(RuntimeError, match="snapshot differs"):
        store.ensure_snapshot({**expected, "input_manifest_fingerprint": "second"})


def test_search_preflight_marks_wikipedia_only_as_degraded(monkeypatch):
    monkeypatch.setattr(cli, "search_web", lambda query, tavily_api_key: {
        "provider": "wikipedia",
        "results": [{"url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
                     "title": "Python", "snippet": "Programming language"}],
    })
    result = cli._search_health_smoke()
    assert result["status"] == "degraded"
    assert result["official_source_found"] is False
    monkeypatch.setattr(cli, "search_web", lambda query, tavily_api_key: {
        "provider": "bing_rss",
        "results": [{"url": "https://docs.python.org/3.12/whatsnew/3.12.html",
                     "title": "What's New in Python 3.12", "snippet": "Official documentation"}],
    })
    assert cli._search_health_smoke()["status"] == "ok"


def test_completed_search_answer_marks_missing_independent_source():
    record = {
        "primary": {"tool_counts": {"web_search_call": 1},
                    "sources": [{"url": "https://en.wikipedia.org/wiki/Example"}]},
        "final": {"tool_counts": {"web_search_call": 0}, "sources": []},
    }
    assert "no independent" in cli._search_evidence_warning(record)
    record["primary"]["sources"].append({"url": "https://example.org/source"})
    assert cli._search_evidence_warning(record) is None


def test_validation_pilot_score_requires_all_twenty_and_forty_percent(monkeypatch, tmp_path, capsys):
    tasks = [Task(f"{index:02d}", f"Question {index}?", 1) for index in range(20)]
    rows = [{"task_id": task.task_id, "Level": 1, "Final answer": "yes", "file_name": ""} for task in tasks]
    fingerprints = {task.task_id: input_fingerprint(task) for task in tasks}
    config = type("Config", (), {"hf_token": ""})()
    snapshot = cli._snapshot_data("validation", tasks, fingerprints, config, "sig")
    store = RunStore(tmp_path)
    store.ensure_snapshot(snapshot)
    manifest = {**snapshot, "task_ids": [task.task_id for task in tasks]}
    manifest_path = tmp_path / "pilot.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    for task in tasks:
        store.put(task.task_id, {
            "task_id": task.task_id, "status": "completed", "answer": "yes" if int(task.task_id) < 8 else "no",
            "run_signature": "sig", "input_fingerprint": fingerprints[task.task_id],
        })
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(cli, "load_gaia_rows", lambda split, token: rows)
    monkeypatch.setattr(cli, "_tasks", lambda kind, config, download: tasks)
    monkeypatch.setattr(cli, "run_signature", lambda config: "sig")
    args = argparse.Namespace(config="unused", split="validation", runs=tmp_path,
                              manifest=str(manifest_path), output=None)
    cli._score(args)
    report = json.loads(capsys.readouterr().out)
    assert report["total"] == 20
    assert report["score_percent"] == 40.0
    assert report["meets_40_percent_gate"] is True
    store.put("00", {"task_id": "00", "status": "error", "run_signature": "sig",
                     "input_fingerprint": fingerprints["00"]})
    cli._score(args)
    report = json.loads(capsys.readouterr().out)
    assert report["pilot_complete"] is False
    assert report["meets_40_percent_gate"] is False
