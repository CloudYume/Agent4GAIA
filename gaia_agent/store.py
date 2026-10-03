"""逐题持久化的私有运行记录。"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, task_id: str) -> Path:
        if not task_id or any(char not in "0123456789abcdef-" for char in task_id.lower()):
            raise ValueError(f"Invalid task id: {task_id}")
        return self.root / f"{task_id}.json"

    def get(self, task_id: str) -> dict[str, Any] | None:
        path = self._path(task_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, task_id: str, record: dict[str, Any]) -> None:
        path = self._path(task_id)
        self._atomic_json(path, record)

    @staticmethod
    def _atomic_json(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            temporary = handle.name
        os.replace(temporary, path)

    @property
    def snapshot_path(self) -> Path:
        return self.root / ".meta" / "run-snapshot.json"

    @property
    def report_path(self) -> Path:
        return self.root / ".meta" / "result.json"

    def snapshot(self) -> dict[str, Any] | None:
        if not self.snapshot_path.exists():
            return None
        return json.loads(self.snapshot_path.read_text(encoding="utf-8"))

    def ensure_snapshot(self, expected: dict[str, Any]) -> dict[str, Any]:
        """Reject a run directory whose code, config or full inputs have changed."""
        previous = self.snapshot()
        if previous is not None:
            comparable = {key: value for key, value in previous.items() if key != "created_at_utc"}
            if comparable != expected:
                raise RuntimeError("Run snapshot differs; use a new --runs directory or restore the original code, config and inputs")
            return previous
        snapshot = {**expected, "created_at_utc": datetime.now(timezone.utc).isoformat()}
        self._atomic_json(self.snapshot_path, snapshot)
        return snapshot

    def write_report(self, report: dict[str, Any]) -> Path:
        self._atomic_json(self.report_path, report)
        return self.report_path
