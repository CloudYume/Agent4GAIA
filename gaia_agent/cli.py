"""题目运行、检查、导出和课程提交的命令行入口。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
from urllib.parse import urlsplit
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from datetime import datetime, timezone

import requests

from .agent import GaiaAgent, is_quota_error
from .config import load_config
from .sources import Task, course_tasks, download_course_attachments, download_gaia_attachment, gaia_tasks, load_gaia_rows
from .scoring import score_records
from .runmeta import dataset_fingerprint, input_fingerprint, input_manifest_fingerprint, run_signature
from .store import RunStore
from .validation import (
    check_export_evidence as _check_export_evidence,
    course_payload,
    evidence_fingerprint as _evidence_fingerprint,
    evidence_issues as _evidence_issues,
    load_review_waivers as _review_waivers,
    validate_official_file,
    TEST_COUNTS,
    validate_stored_record,
    write_official,
)
from .web_tools import search_web


def _tasks(kind: str, config, download: bool) -> list[Task]:
    if kind == "course":
        return course_tasks(config.hf_token or None, download=download)
    return gaia_tasks(kind, config.hf_token or None, download=download)


def _prepare_inputs(tasks: list[Task], split: str, config, *, no_download: bool = False) -> dict[str, str]:
    """Read attachment bytes before deciding whether a saved answer is reusable."""
    attachments = [task for task in tasks if task.file_name]
    if no_download and attachments:
        raise RuntimeError("--no-download cannot verify attachment input fingerprints")
    if attachments and split == "course":
        download_course_attachments(attachments, config.hf_token or None)
    elif attachments:
        for task in attachments:
            download_gaia_attachment(task, split, config.hf_token or None)
    return {task.task_id: input_fingerprint(task) for task in tasks}


def _snapshot_data(split: str, tasks: list[Task], fingerprints: dict[str, str], config, signature: str) -> dict:
    return {
        "schema_version": 1,
        "split": split,
        "run_signature": signature,
        "dataset_fingerprint": dataset_fingerprint(tasks),
        "input_manifest_fingerprint": input_manifest_fingerprint(fingerprints),
        "task_count": len(tasks),
        "level_counts": {str(level): sum(task.level == level for task in tasks) for level in (1, 2, 3)},
        "models": {
            "solver": getattr(config, "solver_model", ""),
            "reviewer": getattr(config, "reviewer_model", ""),
            "judge": getattr(config, "judge_model", ""),
        },
    }


def _evidence_kind(task: Task) -> str:
    question = task.question.lower()
    if "youtube.com/" in question or "youtu.be/" in question:
        return "youtube"
    extension = Path(task.file_name).suffix.lower()
    if extension in {".mp3", ".wav", ".m4a", ".flac", ".ogg"}:
        return "audio"
    if extension in {".mp4", ".webm", ".mov", ".mkv", ".avi"}:
        return "video"
    if extension in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        return "image"
    if extension == ".pdf":
        return "pdf"
    if extension in {".xlsx", ".xls", ".docx", ".pptx"}:
        return "office"
    return "other_attachment" if task.file_name else "text_only"


def _audio_decode_smoke(tasks: list[Task], config) -> str:
    """Exercise the installed PyAV/Whisper decoder without running transcription."""
    if config.transcription_provider != "local":
        return "not_required"
    audio = next((task for task in tasks if _evidence_kind(task) == "audio"), None)
    if audio is None:
        return "no_audio_task"
    try:
        from faster_whisper.audio import decode_audio

        samples = decode_audio(audio.attachment_path)
        if len(samples) == 0:
            raise RuntimeError("decoder returned no audio samples")
    except Exception as exc:
        raise RuntimeError(f"Local ASR decoder smoke test failed ({type(exc).__name__})") from exc
    return "ok"


def _pilot_manifest(tasks: list[Task], snapshot: dict) -> dict:
    """Pick reproducible coverage across levels and available evidence types."""
    if snapshot["split"] == "test":
        if len(tasks) != 301 or Counter(task.level for task in tasks) != Counter(TEST_COUNTS):
            raise ValueError("Test pilot selection requires the complete 2023 GAIA test manifest")
    elif snapshot["split"] != "validation":
        raise ValueError("Pilot selection supports validation and test only")
    quotas = {1: 6, 2: 10, 3: 4}
    if any(sum(task.level == level for task in tasks) < quota for level, quota in quotas.items()):
        raise ValueError("GAIA manifest cannot provide the 6/10/4 stratified pilot")
    selected: set[str] = set()
    counts: Counter[int] = Counter()

    def rank(task: Task) -> str:
        return hashlib.sha256(f"gaia-pilot-v1:{task.task_id}".encode("ascii")).hexdigest()

    ordered = sorted(tasks, key=rank)
    for kind in ("youtube", "audio", "video", "image", "pdf", "office", "other_attachment", "text_only"):
        candidates = [task for task in ordered if _evidence_kind(task) == kind and counts[task.level] < quotas[task.level]]
        if candidates:
            task = candidates[0]
            selected.add(task.task_id)
            counts[task.level] += 1
    for level, quota in quotas.items():
        for task in ordered:
            if counts[level] >= quota:
                break
            if task.level == level and task.task_id not in selected:
                selected.add(task.task_id)
                counts[level] += 1
    chosen = [task for task in tasks if task.task_id in selected]
    if len(chosen) != 20:
        raise RuntimeError(f"Pilot selection expected 20 tasks, got {len(chosen)}")
    return {
        "schema_version": 1,
        "split": snapshot["split"],
        "selection": "stratified-20-v1",
        "run_signature": snapshot["run_signature"],
        "dataset_fingerprint": snapshot["dataset_fingerprint"],
        "input_manifest_fingerprint": snapshot["input_manifest_fingerprint"],
        "task_ids": [task.task_id for task in chosen],
        "level_counts": {str(level): counts[level] for level in (1, 2, 3)},
        "evidence_counts": dict(Counter(_evidence_kind(task) for task in chosen)),
    }


def _search_health_smoke() -> dict:
    """Classify weak search separately so a validation pilot can still proceed."""
    try:
        result = search_web(
            "Python 3.12 release notes official documentation",
            tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
        )
    except (RuntimeError, ValueError) as exc:
        return {"status": "degraded", "reason": f"search_unavailable:{type(exc).__name__}"}
    official = any(
        (urlsplit(item["url"]).hostname or "").lower() in {"python.org", "www.python.org", "docs.python.org"}
        and "3.12" in (item.get("title", "") + " " + item.get("snippet", ""))
        for item in result["results"]
    )
    return {
        "status": "ok" if official else "degraded",
        "provider": result["provider"],
        "official_source_found": official,
        "reason": "" if official else "no_relevant_independent_source",
    }


def _search_evidence_warning(record: dict) -> str | None:
    results = [result for result in (record.get("primary"), record.get("final")) if isinstance(result, dict)]
    if not any((result.get("tool_counts") or {}).get("web_search_call") for result in results):
        return None
    sources = [source for result in results for source in (result.get("sources") or [])]
    independent = any(
        (host := (urlsplit(str(source.get("url", ""))).hostname or "").lower())
        and host not in {"wikipedia.org", "www.wikipedia.org", "en.wikipedia.org"}
        and not host.endswith(".wikipedia.org")
        for source in sources if isinstance(source, dict)
    )
    return None if independent else "Web search supplied no independent non-Wikipedia source; verify evidence before export"


def _manifest_tasks(path: str, tasks: list[Task], snapshot: dict) -> list[Task]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or manifest.get("split") != snapshot["split"]
            or any(manifest.get(key) != snapshot[key] for key in (
                "run_signature", "dataset_fingerprint", "input_manifest_fingerprint"
            ))):
        raise RuntimeError("Task manifest differs from the current run snapshot")
    ids = manifest.get("task_ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(item, str) for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("Task manifest needs unique nonempty task_ids")
    known = {task.task_id for task in tasks}
    if not set(ids) <= known:
        raise ValueError("Task manifest includes unknown task IDs")
    return [task for task in tasks if task.task_id in set(ids)]


def _result_report(tasks: list[Task], store: RunStore, snapshot: dict) -> dict:
    statuses: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    failed_ids = []
    evidence_ids = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    for task in tasks:
        record = store.get(task.task_id)
        if record is None:
            status = "pending"
        elif record.get("run_signature") != snapshot["run_signature"]:
            status = "signature_mismatch"
        elif record.get("input_fingerprint") != snapshot["input_fingerprints"][task.task_id]:
            status = "input_mismatch"
        else:
            status = str(record.get("status") or "unknown")
        statuses[status] += 1
        if status != "completed":
            failed_ids.append(task.task_id)
        if not record or status != "completed":
            continue
        issues = _evidence_issues(record)
        if record.get("review_error"):
            issues.add("review_error")
        if record.get("adjudication_error"):
            issues.add("adjudication_error")
        if issues:
            evidence_ids.append(task.task_id)
            issue_counts.update(issues)
        seen_response_ids = set()
        for stage in ("primary", "review", "final"):
            result = record.get(stage)
            if not isinstance(result, dict):
                continue
            response_id = result.get("response_id") or ("duplicate", json.dumps(result, sort_keys=True, default=str))
            if response_id in seen_response_ids:
                continue
            seen_response_ids.add(response_id)
            stage_usage = result.get("usage") or {}
            for key in usage:
                usage[key] += int(stage_usage.get(key, 0) or 0)
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": snapshot["split"],
        "run_signature": snapshot["run_signature"],
        "dataset_fingerprint": snapshot["dataset_fingerprint"],
        "input_manifest_fingerprint": snapshot["input_manifest_fingerprint"],
        "task_count": len(tasks),
        "status_counts": dict(statuses),
        "evidence_issue_counts": dict(issue_counts),
        "token_usage_completed_records": usage,
        "unfinished_task_ids": failed_ids,
        "evidence_review_task_ids": evidence_ids,
        "strict_ready_without_waivers": statuses["completed"] == len(tasks) and not evidence_ids,
    }


def _preflight(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    config.require(config.solver_provider)
    if config.review_mode != "off":
        config.require(config.reviewer_provider)
        config.require(config.judge_provider)
    tasks = _tasks(args.split, config, download=False)
    fingerprints = _prepare_inputs(tasks, args.split, config)
    audio_decode_smoke = _audio_decode_smoke(tasks, config)
    search_health = _search_health_smoke()
    signature = run_signature(config)
    snapshot = _snapshot_data(args.split, tasks, fingerprints, config, signature)
    store = RunStore(args.runs)
    store.ensure_snapshot(snapshot)
    pilot = _pilot_manifest(tasks, snapshot)
    pilot_path = Path(args.pilot_output) if args.pilot_output else store.root / ".meta" / "pilot-20.json"
    RunStore._atomic_json(pilot_path, pilot)
    report = {
        **snapshot,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "attachment_count": sum(bool(task.file_name) for task in tasks),
        "youtube_task_count": sum(_evidence_kind(task) == "youtube" for task in tasks),
        "credentials_present": {
            "solver": True, "judge": True,
            "reviewer": config.review_mode == "off" or bool(config.anthropic_key),
            "huggingface": bool(config.hf_token),
        },
        "dependencies_present": {
            "yt_dlp": importlib.util.find_spec("yt_dlp") is not None,
            "imageio_ffmpeg": importlib.util.find_spec("imageio_ffmpeg") is not None,
            "faster_whisper": importlib.util.find_spec("faster_whisper") is not None,
            "ffmpeg": bool(shutil.which("ffmpeg")),
        },
        "youtube_browser_cookies_configured": bool(config.youtube_cookies_from_browser),
        "audio_decode_smoke": audio_decode_smoke,
        "search_health": search_health,
        "pilot_manifest": str(pilot_path),
    }
    RunStore._atomic_json(store.root / ".meta" / "preflight.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _status(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    tasks = _tasks(args.split, config, download=False)
    fingerprints = _prepare_inputs(tasks, args.split, config)
    signature = run_signature(config)
    snapshot = _snapshot_data(args.split, tasks, fingerprints, config, signature)
    store = RunStore(args.runs)
    report = _result_report(tasks, store, {**snapshot, "input_fingerprints": fingerprints})
    report["snapshot_matches"] = store.snapshot() is not None and {
        key: value for key, value in store.snapshot().items() if key != "created_at_utc"
    } == snapshot
    report["strict_ready_without_waivers"] &= report["snapshot_matches"]
    output = Path(args.output) if args.output else store.report_path
    RunStore._atomic_json(output, report)
    print(json.dumps({"result": str(output), "status_counts": report["status_counts"],
                      "evidence_issue_counts": report["evidence_issue_counts"],
                      "strict_ready_without_waivers": report["strict_ready_without_waivers"]}, indent=2))


def _run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    signature = run_signature(config)
    all_tasks = _tasks(args.split, config, download=False)
    fingerprints = _prepare_inputs(all_tasks, args.split, config, no_download=args.no_download)
    store = RunStore(args.runs)
    snapshot = _snapshot_data(args.split, all_tasks, fingerprints, config, signature)
    if args.split in {"test", "validation"}:
        store.ensure_snapshot(snapshot)
    tasks = all_tasks
    if getattr(args, "manifest", None):
        tasks = _manifest_tasks(args.manifest, tasks, snapshot)
    if args.task_id:
        tasks = [task for task in tasks if task.task_id == args.task_id]
        if not tasks:
            raise ValueError(f"Unknown task ID: {args.task_id}")
    agent = GaiaAgent(config)
    for task in tasks:
        old = store.get(task.task_id)
        if old and old.get("run_signature") != signature and not args.force:
            raise RuntimeError("Run settings changed. Use a new --runs directory or rerun all tasks with --force.")
    pending = []
    for task in tasks:
        record = store.get(task.task_id) or {}
        if (args.force or record.get("input_fingerprint") != fingerprints[task.task_id]
                or record.get("status") != "completed"
                or (args.retry_review_errors and record.get("review_error"))):
            pending.append(task)
    if args.task_id and not pending:
        raise ValueError(f"No pending task with ID {args.task_id}")
    if args.limit:
        pending = pending[: args.limit]
    print(f"Loaded {len(tasks)} tasks; running {len(pending)} pending tasks")

    def save_record(task: Task, record: dict) -> None:
        stored = dict(record)
        if stored.get("status") == "completed":
            warning = _search_evidence_warning(stored)
            if warning:
                stored["warnings"] = list(stored.get("warnings") or [])
                if warning not in stored["warnings"]:
                    stored["warnings"].append(warning)
        stored.update({
            "task_id": task.task_id, "question": task.question, "level": task.level,
            "run_signature": signature, "input_fingerprint": fingerprints[task.task_id],
        })
        store.put(task.task_id, stored)

    def solve(task: Task):
        try:
            previous = None if args.force else store.get(task.task_id)
            if previous and previous.get("input_fingerprint") != fingerprints[task.task_id]:
                previous = None
            return task, agent.solve(task, previous=previous, checkpoint=lambda stage: save_record(task, stage))
        except Exception as exc:
            status = "quota_exhausted" if is_quota_error(exc) else "error"
            partial = store.get(task.task_id) or {}
            if args.force or partial.get("input_fingerprint") != fingerprints[task.task_id]:
                partial = {}
            return task, {**partial, "status": status, "error": f"{type(exc).__name__}: {exc}"}

    def checkpoint(future) -> dict:
        task, record = future.result()
        save_record(task, record)
        print(f"{task.task_id} {record['status']}: {record.get('answer', record.get('error', ''))}")
        return record

    quota_exhausted = False
    breaker_reason = ""
    error_kinds: Counter[str] = Counter()
    recent_errors: deque[bool] = deque(maxlen=20)
    completed_count = 0
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        remaining = iter(pending)
        active = set()
        try:
            for _ in range(min(config.workers, len(pending))):
                active.add(executor.submit(solve, next(remaining)))
            while active:
                finished, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in finished:
                    record = checkpoint(future)
                    active.remove(future)
                    completed_count += 1
                    status = record.get("status")
                    quota_exhausted |= status == "quota_exhausted"
                    failed = status in {"error", "needs_review", "needs_adjudication", "budget_exhausted"}
                    recent_errors.append(failed)
                    if failed:
                        detail = record.get("error") or record.get("review_error") or record.get("adjudication_error") or status
                        error_kinds[":".join(str(detail).split(":")[:2]).strip()] += 1
                    if any(count >= 3 for count in error_kinds.values()) and completed_count <= 12:
                        breaker_reason = "Three similar errors occurred in the first 12 tasks"
                    if len(recent_errors) == 20 and sum(recent_errors) >= 8:
                        breaker_reason = "Eight of the last 20 tasks failed"
                while not (quota_exhausted or breaker_reason) and len(active) < config.workers:
                    task = next(remaining, None)
                    if task is not None:
                        active.add(executor.submit(solve, task))
                    else:
                        break
        except KeyboardInterrupt:
            print("Interrupted; saving results from active tasks before exit")
            for future in as_completed(active):
                checkpoint(future)
            store.write_report(_result_report(all_tasks, store, {**snapshot, "input_fingerprints": fingerprints}))
            raise
    store.write_report(_result_report(all_tasks, store, {**snapshot, "input_fingerprints": fingerprints}))
    if quota_exhausted:
        raise RuntimeError("API quota exhausted; completed task records were saved and no new tasks were scheduled")
    if breaker_reason:
        raise RuntimeError(f"Batch circuit breaker opened: {breaker_reason}; completed task records were saved")


def _records(
    runs: str | Path, tasks: list[Task], signature: str,
    fingerprints: dict[str, str] | None = None,
) -> list[dict]:
    store = RunStore(runs)
    records = []
    for task in tasks:
        record = store.get(task.task_id)
        expected_input = fingerprints[task.task_id] if fingerprints is not None else (record or {}).get("input_fingerprint")
        validate_stored_record(record, task.task_id, expected_input, signature)
        records.append(record)
    return records


def _export(args: argparse.Namespace) -> None:
    if args.fill_failures:
        raise ValueError("--fill-failures cannot produce a strict official submission; rerun failed tasks")
    if args.accept_partial_evidence:
        raise ValueError("Use per-task --review-waivers after inspecting the evidence")
    if getattr(args, "review_waivers", None) and args.split != "test":
        raise ValueError("--review-waivers is available only for official test export")
    if args.split == "validation":
        raise ValueError("Official JSONL export is for the test split only")
    config = load_config(args.config)
    tasks = _tasks(args.split, config, download=False)
    fingerprints = _prepare_inputs(tasks, args.split, config)
    if args.split == "test":
        expected_snapshot = _snapshot_data(args.split, tasks, fingerprints, config, run_signature(config))
        saved_snapshot = RunStore(args.runs).snapshot()
        if saved_snapshot is None or {
            key: value for key, value in saved_snapshot.items() if key != "created_at_utc"
        } != expected_snapshot:
            raise RuntimeError("Official test run snapshot is missing or differs; rerun preflight in the original run directory")
    records = _records(args.runs, tasks, run_signature(config), fingerprints)
    if args.split == "test":
        warning_count = sum(bool(row.get("warnings")) for row in records)
        review_error_count = sum(bool(row.get("review_error")) for row in records)
        unresolved_count = sum(bool(row.get("adjudication_error")) for row in records)
        _check_export_evidence(records, _review_waivers(getattr(args, "review_waivers", None)))
    if args.split == "course":
        payload = course_payload(args.username, args.agent_code, records)
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        write_official(
            args.output,
            [{"task_id": row["task_id"], "model_answer": row["answer"]} for row in records],
            {task.task_id: task.level for task in tasks},
        )
        print(
            f"Evidence: warnings={warning_count}, review_errors={review_error_count}, "
            f"unresolved={unresolved_count}, blank_answers={sum(not row['answer'] for row in records)}"
        )
    print(f"Wrote {args.output}")


def _check(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    tasks = gaia_tasks(args.split, config.hf_token or None, download=False)
    levels = {task.task_id: task.level for task in tasks}
    rows = validate_official_file(args.file, levels)
    print(f"Valid official submission: {len(rows)} records; levels={ {level: list(levels.values()).count(level) for level in (1, 2, 3)} }")


def _submit_course(args: argparse.Namespace) -> None:
    payload = json.loads(Path(args.payload).read_text(encoding="utf-8"))
    expected_ids = {task.task_id for task in course_tasks(download=False)}
    payload = course_payload(
        payload["username"],
        payload["agent_code"],
        [{"task_id": row["task_id"], "answer": row["submitted_answer"]} for row in payload["answers"]],
        expected_ids=expected_ids,
    )
    response = requests.post("https://agents-course-unit4-scoring.hf.space/submit", json=payload, timeout=120)
    response.raise_for_status()
    print(response.text)


def _score(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    rows = load_gaia_rows("validation", config.hf_token or None)
    if getattr(args, "manifest", None):
        if args.split != "validation":
            raise ValueError("--manifest scoring applies to validation only")
        tasks = _tasks("validation", config, download=False)
        fingerprints = _prepare_inputs(tasks, "validation", config)
        snapshot = _snapshot_data("validation", tasks, fingerprints, config, run_signature(config))
        store = RunStore(args.runs)
        saved_snapshot = store.snapshot()
        if saved_snapshot is None or {
            key: value for key, value in saved_snapshot.items() if key != "created_at_utc"
        } != snapshot:
            raise RuntimeError("Validation pilot snapshot is missing or differs")
        chosen = _manifest_tasks(args.manifest, tasks, snapshot)
        if len(chosen) != 20:
            raise ValueError("Validation pilot scoring requires exactly 20 task IDs")
        chosen_ids = {task.task_id for task in chosen}
        rows = [row for row in rows if str(row["task_id"]) in chosen_ids]
    if args.split == "course":
        course_ids = {task.task_id for task in course_tasks(download=False)}
        rows = [row for row in rows if str(row["task_id"]) in course_ids]
        if len(rows) != 20:
            raise ValueError(f"Expected 20 course questions in validation metadata, got {len(rows)}")
    store = RunStore(args.runs)
    records = {str(row["task_id"]): store.get(str(row["task_id"])) for row in rows}
    if getattr(args, "manifest", None):
        for task_id, record in records.items():
            if record and record.get("run_signature") != snapshot["run_signature"]:
                records[task_id] = {"status": "signature_mismatch"}
            elif record and record.get("input_fingerprint") != fingerprints[task_id]:
                records[task_id] = {"status": "input_mismatch"}
    report = score_records(rows, records, mode="course_exact" if args.split == "course" else "official")
    if getattr(args, "manifest", None):
        report["pilot_complete"] = report["run_status"].get("completed", 0) == 20
        report["meets_40_percent_gate"] = report["pilot_complete"] and report["score_percent"] >= 40.0
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _doctor(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    print(
        json.dumps(
            {
                "solver": config.solver_model,
                "solver_provider": config.solver_provider,
                "openai_base_url": config.openai_base_url or "official default",
                "deepseek_base_url": config.deepseek_base_url,
                "reviewer": config.reviewer_model,
                "anthropic_base_url": config.anthropic_base_url or "official default",
                "judge": config.judge_model,
                "judge_provider": config.judge_provider,
                "review_mode": config.review_mode,
                "code_interpreter_enabled": config.code_interpreter_enabled,
                "search_context_size": config.search_context_size,
                "max_tool_calls": config.max_tool_calls,
                "max_output_tokens": config.max_output_tokens,
                "openai_key_set": bool(config.openai_key),
                "deepseek_key_set": bool(config.deepseek_key),
                "anthropic_key_set": bool(config.anthropic_key),
                "hf_token_set": bool(config.hf_token),
                "youtube_browser_cookies_configured": bool(config.youtube_cookies_from_browser),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml")
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight", help="Verify GAIA access and write a fixed 20-task pilot manifest")
    preflight.add_argument("split", choices=["validation", "test"])
    preflight.add_argument("--runs", default=".runs/validation-pilot")
    preflight.add_argument("--pilot-output", help="Defaults to RUNS/.meta/pilot-20.json")
    preflight.set_defaults(func=_preflight)
    run = sub.add_parser("run")
    run.add_argument("split", choices=["course", "validation", "test"])
    run.add_argument("--runs", default=".runs/records")
    run.add_argument("--no-download", action="store_true")
    run.add_argument("--force", action="store_true")
    run.add_argument("--retry-review-errors", action="store_true")
    run.add_argument("--task-id")
    run.add_argument("--manifest", help="Run only task IDs from a preflight manifest")
    run.add_argument("--limit", type=int)
    run.set_defaults(func=_run)
    status = sub.add_parser("status", help="Write a private result.json progress report")
    status.add_argument("split", choices=["course", "validation", "test"])
    status.add_argument("--runs", default=".runs/records")
    status.add_argument("--output", help="Defaults to RUNS/.meta/result.json")
    status.set_defaults(func=_status)
    export = sub.add_parser("export")
    export.add_argument("split", choices=["course", "validation", "test"])
    export.add_argument("--runs", default=".runs/records")
    export.add_argument("--output", required=True)
    export.add_argument("--username", default=os.getenv("HF_USERNAME", ""))
    export.add_argument("--agent-code", default="")
    export.add_argument("--fill-failures", action="store_true")
    export.add_argument("--accept-partial-evidence", action="store_true")
    export.add_argument("--review-waivers", help="JSON map of reviewed per-task evidence exceptions")
    export.set_defaults(func=_export)
    check = sub.add_parser("check-official")
    check.add_argument("file")
    check.add_argument("--split", choices=["test"], default="test")
    check.set_defaults(func=_check)
    submit = sub.add_parser("submit-course")
    submit.add_argument("payload")
    submit.set_defaults(func=_submit_course)
    score = sub.add_parser("score-validation")
    score.add_argument("split", choices=["course", "validation"])
    score.add_argument("--runs", default=".runs/records")
    score.add_argument("--manifest", help="Score exactly the 20 validation pilot task IDs")
    score.add_argument("--output")
    score.set_defaults(func=_score)
    doctor = sub.add_parser("doctor")
    doctor.set_defaults(func=_doctor)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
