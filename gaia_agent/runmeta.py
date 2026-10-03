"""运行配置指纹，防止不同模型或提示的答案混入同一批次。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .agent import JUDGE_PROMPT, REVIEW_PROMPT, _solver_prompt
from .config import Config
from .sources import Task


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    directory = Path(__file__).resolve().parent
    for name in (
        "agent.py", "deepseek.py", "web_tools.py", "reviewer.py", "attachments.py",
        "media.py", "sources.py", "config.py", "cli.py", "validation.py",
        "store.py", "scoring.py", "runmeta.py",
        "contracts.py", "orchestrator.py", "memory.py",
        "tools/__init__.py", "tools/data.py", "tools/vision.py",
    ):
        digest.update(name.encode("ascii"))
        digest.update(b"\x00")
        digest.update((directory / name).read_bytes())
    return digest.hexdigest()[:16]


def input_fingerprint(task: Task) -> str:
    """Bind a task record to the exact question and local attachment bytes."""
    attachment_hash = None
    if task.file_name:
        if not task.attachment_path:
            raise RuntimeError(task.attachment_error or f"Missing attachment for input fingerprint: {task.task_id}")
        digest = hashlib.sha256()
        with Path(task.attachment_path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        attachment_hash = digest.hexdigest()
    details = {
        "version": 1,
        "task_id": task.task_id,
        "question": task.question,
        "level": task.level,
        "file_name": task.file_name,
        "repo_file": task.repo_file,
        "attachment_sha256": attachment_hash,
    }
    return hashlib.sha256(json.dumps(details, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def dataset_fingerprint(tasks: list[Task]) -> str:
    """Bind a run to the ordered task manifest without storing question text."""
    rows = [
        {
            "task_id": task.task_id, "question": task.question, "level": task.level,
            "file_name": task.file_name, "repo_file": task.repo_file,
        }
        for task in tasks
    ]
    payload = json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def input_manifest_fingerprint(fingerprints: dict[str, str]) -> str:
    payload = json.dumps(fingerprints, sort_keys=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def run_signature(config: Config) -> str:
    details = {
        "version": 7,
        "implementation": _implementation_hash(),
        "solver_provider": config.solver_provider,
        "solver_model": config.solver_model,
        "reviewer_provider": config.reviewer_provider,
        "reviewer_model": config.reviewer_model,
        "judge_provider": config.judge_provider,
        "judge_model": config.judge_model,
        "openai_base_url": config.openai_base_url,
        "deepseek_base_url": config.deepseek_base_url,
        "deepseek_search_provider": "tavily" if os.getenv("TAVILY_API_KEY") else "bing_rss",
        "anthropic_base_url": config.anthropic_base_url,
        "reasoning_effort": config.reasoning_effort,
        "code_interpreter_enabled": config.code_interpreter_enabled,
        "search_context_size": config.search_context_size,
        "max_tool_calls": config.max_tool_calls,
        "max_output_tokens": config.max_output_tokens,
        "review_mode": config.review_mode,
        "confidence_threshold": config.confidence_threshold,
        "transcription_model": config.transcription_model,
        "transcription_provider": config.transcription_provider,
        "local_asr_model": config.local_asr_model,
        "youtube_cookies_from_browser": config.youtube_cookies_from_browser,
        "prompts": [_solver_prompt(config), REVIEW_PROMPT, JUDGE_PROMPT],
    }
    payload = json.dumps(details, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
