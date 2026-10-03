"""On-demand solver, blind reviewer, and dispute adjudicator roles."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from .contracts import BudgetExceeded, StageResult, TaskContext


def _charge_failed_call(context: TaskContext, role: str, started: float, exc: Exception) -> None:
    diagnostics = getattr(exc, "diagnostics", None)
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    payload = {
        "diagnostics": diagnostics,
        "attempts": getattr(exc, "attempts", 1) if role == "reviewer" else None,
        "usage": getattr(exc, "usage", {}),
        "error_type": type(exc).__name__,
    }
    stage = StageResult.from_payload(role, payload, time.monotonic() - started)
    context.memory.add_model_result(stage)
    try:
        context.budget.charge(stage)
    except BudgetExceeded:
        pass


def needs_review(context: TaskContext, primary: dict[str, Any], mode: str, confidence_threshold: float) -> bool:
    if mode == "always":
        return True
    if mode != "adaptive":
        return False
    quality = primary.get("evidence_quality") or {}
    diagnostics = primary.get("diagnostics") or {}
    return (
        context.task.level >= 2
        or bool(context.task.file_name)
        or bool(context.prepared.review_images)
        or bool(context.prepared.warnings)
        or float(primary.get("confidence", 0)) < confidence_threshold
        or quality.get("failed", 0) > 0
        or quality.get("tool_failures", 0) > 0
        or (quality.get("search_only", 0) > 0 and quality.get("fetched", 0) == 0)
        or bool(diagnostics.get("evidence_degraded"))
    )


@dataclass
class SolverRole:
    model_call: Callable[..., dict[str, Any]]
    provider: str
    model: str
    instructions: str
    name: str = "solver"

    def run(self, context: TaskContext, primary: dict[str, Any] | None = None,
            review: dict[str, Any] | None = None) -> StageResult:
        context.budget.ensure(self.name)
        started = time.monotonic()
        try:
            payload = self.model_call(
                provider=self.provider, model=self.model, instructions=self.instructions,
                content=context.prepared.content,
                attachment_path=context.task.attachment_path,
                remaining_model_calls=context.budget.remaining_model_calls,
            )
        except Exception as exc:
            _charge_failed_call(context, self.name, started, exc)
            raise
        stage = StageResult.from_payload(self.name, payload, time.monotonic() - started)
        context.memory.add_model_result(stage)
        context.budget.charge(stage)
        return stage


@dataclass
class ReviewerRole:
    review_call: Callable[..., dict[str, Any]]
    name: str = "reviewer"

    def run(self, context: TaskContext, primary: dict[str, Any] | None = None,
            review: dict[str, Any] | None = None) -> StageResult:
        if primary is None:
            raise ValueError("Reviewer requires a primary candidate")
        # The Anthropic reviewer can use three web turns and two recovery turns.
        context.budget.ensure(self.name, required_calls=5)
        started = time.monotonic()
        try:
            payload = self.review_call(context.task, primary, context.prepared)
        except Exception as exc:
            _charge_failed_call(context, self.name, started, exc)
            raise
        stage = StageResult.from_payload(self.name, payload, time.monotonic() - started)
        context.memory.add_model_result(stage)
        context.budget.charge(stage)
        return stage


def _dispute_brief(primary: dict[str, Any], review: dict[str, Any]) -> str:
    sources = [
        {"url": source.get("url", ""), "status": source.get("status", ""),
         "excerpt": str(source.get("excerpt") or source.get("snippet") or "")[:800]}
        for source in primary.get("sources", []) if isinstance(source, dict)
    ][:5]
    brief = {
        "primary": {"answer": primary.get("answer"), "confidence": primary.get("confidence"),
                    "evidence": primary.get("evidence", [])[:5], "sources": sources},
        "review": {"proposed_answer": review.get("proposed_answer"), "verdict": review.get("verdict"),
                   "certainty": review.get("certainty"), "reason": str(review.get("reason") or "")[:800],
                   "evidence": review.get("evidence", [])[:5], "citations": review.get("citations", [])[:5]},
    }
    return "Evaluate this disagreement using the original task and evidence: " + json.dumps(brief, ensure_ascii=False)


@dataclass
class JudgeRole:
    model_call: Callable[..., dict[str, Any]]
    provider: str
    model: str
    instructions: str
    name: str = "judge"

    def run(self, context: TaskContext, primary: dict[str, Any] | None = None,
            review: dict[str, Any] | None = None) -> StageResult:
        if primary is None or review is None:
            raise ValueError("Judge requires both candidates")
        context.budget.ensure(self.name)
        started = time.monotonic()
        try:
            payload = self.model_call(
                provider=self.provider, model=self.model, instructions=self.instructions,
                content=context.prepared.content + [{"type": "input_text", "text": _dispute_brief(primary, review)}],
                attachment_path=context.task.attachment_path,
                remaining_model_calls=context.budget.remaining_model_calls,
            )
        except Exception as exc:
            _charge_failed_call(context, self.name, started, exc)
            raise
        stage = StageResult.from_payload(self.name, payload, time.monotonic() - started)
        context.memory.add_model_result(stage)
        context.budget.charge(stage)
        return stage
