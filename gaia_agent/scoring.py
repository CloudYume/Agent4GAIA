"""公开验证集的本地评分与难度分组统计。"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _number(value: str) -> float:
    try:
        return float(value.replace("$", "").replace("%", "").replace(",", ""))
    except ValueError:
        return float("inf")


def _text(value: str, remove_punctuation: bool = True) -> str:
    value = re.sub(r"\s", "", value).lower()
    return value.translate(str.maketrans("", "", string.punctuation)) if remove_punctuation else value


def question_scorer(answer: str, ground_truth: str) -> bool:
    if _is_float(ground_truth):
        return _number(answer) == float(ground_truth)
    if "," in ground_truth or ";" in ground_truth:
        expected = re.split("[,;]", ground_truth)
        actual = re.split("[,;]", answer)
        if len(expected) != len(actual):
            return False
        return all(
            _number(a) == float(e) if _is_float(e) else _text(a, False) == _text(e, False)
            for a, e in zip(actual, expected)
        )
    return _text(answer) == _text(ground_truth)


def score_records(rows: list[dict[str, Any]], records: dict[str, dict[str, Any]], *, mode: str = "official") -> dict[str, Any]:
    if mode not in {"official", "course_exact"}:
        raise ValueError(f"Unknown scoring mode: {mode}")
    by_level: dict[int, Counter] = {1: Counter(), 2: Counter(), 3: Counter()}
    by_attachment: dict[str, Counter] = {"with_file": Counter(), "without_file": Counter()}
    by_status: Counter = Counter()
    failed_ids = []
    for row in rows:
        task_id = str(row["task_id"])
        level = int(row["Level"])
        record = records.get(task_id) or {}
        by_status[record.get("status") or "not_run"] += 1
        answer = record.get("answer", "") if record.get("status") == "completed" else ""
        truth = str(row["Final answer"])
        correct = bool(answer) and (answer == truth if mode == "course_exact" else question_scorer(answer, truth))
        by_level[level]["total"] += 1
        by_level[level]["correct"] += int(correct)
        group = "with_file" if row.get("file_name") else "without_file"
        by_attachment[group]["total"] += 1
        by_attachment[group]["correct"] += int(correct)
        if not correct:
            failed_ids.append(task_id)
    total = len(rows)
    correct = sum(counts["correct"] for counts in by_level.values())
    return {
        "total": total,
        "correct": correct,
        "score_percent": round(100 * correct / total, 2) if total else 0.0,
        "scoring_mode": mode,
        "run_status": dict(by_status),
        "by_level": {str(level): dict(counts) for level, counts in by_level.items()},
        "by_attachment": {name: dict(counts) for name, counts in by_attachment.items()},
        "failed_task_ids": failed_ids,
    }
