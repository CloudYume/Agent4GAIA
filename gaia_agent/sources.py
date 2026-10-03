"""课程题目与受限 GAIA 附件的读取。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow.parquet as pq
import requests
from huggingface_hub import hf_hub_download

COURSE_URL = "https://agents-course-unit4-scoring.hf.space"
DATASET_ID = "gaia-benchmark/GAIA"

_MAGIC_PREFIXES = {
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".gif": (b"GIF87a", b"GIF89a"),
    ".pdf": (b"%PDF-",),
    ".xlsx": (b"PK\x03\x04",),
    ".docx": (b"PK\x03\x04",),
    ".pptx": (b"PK\x03\x04",),
    ".zip": (b"PK\x03\x04",),
    ".webm": (b"\x1a\x45\xdf\xa3",),
    ".wav": (b"RIFF",),
}


@dataclass
class Task:
    task_id: str
    question: str
    level: int
    file_name: str = ""
    attachment_path: str = ""
    attachment_error: str = ""
    repo_file: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Task":
        return cls(**value)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _check_course_attachment(content: bytes, file_name: str, content_type: str = "") -> None:
    if not content:
        raise ValueError("empty attachment response")
    extension = Path(file_name).suffix.lower()
    start = content.lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    if extension != ".html" and (
        content_type.lower().split(";", 1)[0].strip() == "text/html"
        or start.startswith((b"<!doctype html", b"<html", b"<head", b"<body"))
    ):
        raise ValueError("HTML page returned instead of attachment")
    if extension != ".json" and content_type.lower().split(";", 1)[0].strip() == "application/json":
        raise ValueError("JSON response returned instead of attachment")
    if extension in _MAGIC_PREFIXES and not content.startswith(_MAGIC_PREFIXES[extension]):
        raise ValueError(f"attachment content does not match {extension}")
    if extension == ".mp3" and not (
        content.startswith(b"ID3") or (len(content) > 1 and content[0] == 0xFF and content[1] & 0xE0 == 0xE0)
    ):
        raise ValueError("attachment content does not match .mp3")
    if extension in {".mp4", ".mov"} and (len(content) < 8 or content[4:8] != b"ftyp"):
        raise ValueError(f"attachment content does not match {extension}")
    if extension == ".wav" and (len(content) < 12 or content[8:12] != b"WAVE"):
        raise ValueError("attachment content does not match .wav")
    if extension in {".py", ".txt", ".md", ".csv"} and b"\x00" in content[:4096]:
        raise ValueError(f"binary response returned for {extension} attachment")


def _repo_file(row: dict[str, Any], split: str) -> str:
    raw = _text(row.get("file_path")) or _text(row.get("file_name"))
    if not raw:
        return ""
    path = PurePosixPath(raw.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe attachment path: {raw}")
    if path.parts[:2] == ("2023", split):
        return path.as_posix()
    if len(path.parts) == 1:
        return f"2023/{split}/{path.name}"
    raise ValueError(f"Unexpected attachment path: {raw}")


def load_gaia_rows(split: str, token: str | None = None) -> list[dict[str, Any]]:
    if split not in {"test", "validation"}:
        raise ValueError("GAIA split must be test or validation")
    try:
        metadata = hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=f"2023/{split}/metadata.parquet",
            token=token,
        )
    except Exception as exc:
        raise RuntimeError(
            "GAIA is gated. Accept its dataset terms and configure HF_TOKEN "
            "or run `hf auth login` before fetching questions."
        ) from exc
    return pq.read_table(metadata).to_pylist()


def gaia_tasks(split: str, token: str | None = None, download: bool = True) -> list[Task]:
    tasks = []
    for row in load_gaia_rows(split, token):
        file_name = _text(row.get("file_name"))
        repo_file = _repo_file(row, split) if file_name or row.get("file_path") else ""
        task = Task(
            task_id=_text(row.get("task_id")),
            question=_text(row.get("Question") or row.get("question")),
            level=int(row.get("Level") or row.get("level")),
            file_name=Path(file_name).name if file_name else (PurePosixPath(repo_file).name if repo_file else ""),
            repo_file=repo_file,
        )
        if repo_file and download:
            download_gaia_attachment(task, split, token, repo_file)
        tasks.append(task)
    return tasks


def download_gaia_attachment(task: Task, split: str, token: str | None, repo_file: str | None = None) -> None:
    if not task.file_name:
        return
    filename = repo_file or task.repo_file or _repo_file({"file_name": task.file_name}, split)
    try:
        task.attachment_path = hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=filename,
            token=token,
        )
    except Exception as exc:
        task.attachment_error = f"Cannot download {filename}: {exc}"


def course_tasks(token: str | None = None, download: bool = True) -> list[Task]:
    response = requests.get(f"{COURSE_URL}/questions", timeout=30)
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list):
        raise ValueError("Course API returned a non-list question payload")
    tasks = [
        Task(
            task_id=_text(row["task_id"]),
            question=_text(row["question"]),
            level=int(row["Level"]),
            file_name=Path(_text(row.get("file_name"))).name,
        )
        for row in rows
    ]
    if download:
        download_course_attachments(tasks, token)
    return tasks


def download_course_attachments(tasks: list[Task], token: str | None = None) -> None:
    validation = None
    for task in tasks:
        if not task.file_name:
            continue
        try:
            response = requests.get(f"{COURSE_URL}/files/{task.task_id}", timeout=60)
            response.raise_for_status()
            _check_course_attachment(
                response.content,
                task.file_name,
                getattr(response, "headers", {}).get("Content-Type", ""),
            )
            if not task.task_id or any(char not in "0123456789abcdef-" for char in task.task_id.lower()):
                raise ValueError("invalid task ID for attachment storage")
            target = Path(".runs") / "course-files" / task.task_id / task.file_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(response.content)
            task.attachment_path = str(target.resolve())
            continue
        except (requests.RequestException, ValueError) as exc:
            course_error = f"Course attachment endpoint failed: {exc}"
        try:
            if validation is None:
                validation = {str(row["task_id"]): row for row in load_gaia_rows("validation", token)}
            row = validation[task.task_id]
            task.attachment_path = hf_hub_download(
                repo_id=DATASET_ID,
                repo_type="dataset",
                filename=_repo_file(row, "validation"),
                token=token,
            )
        except Exception as exc:
            task.attachment_error = f"{course_error}; GAIA fallback failed: {exc}"
