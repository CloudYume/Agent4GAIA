"""Typed boundaries for one GAIA task and its cooperating roles."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from .attachments import PreparedInput
    from .memory import TaskMemory
    from .sources import Task


@dataclass(frozen=True)
class Evidence:
    source_id: str
    kind: str
    locator: str
    status: str
    excerpt: str = ""
    sha256: str = ""
    truncated: bool = False
    warning: str = ""

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Observation:
    asset_id: str
    kind: str
    label: str
    locator: str
    timestamp: float | None = None
    text: str = ""
    uncertain: bool = False
    question: str = ""
    region: str = ""
    cache_hit: bool = False

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StageResult:
    role: str
    payload: dict[str, Any]
    model_calls: int
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float

    @classmethod
    def from_payload(cls, role: str, payload: dict[str, Any], elapsed_seconds: float = 0) -> StageResult:
        usage = payload.get("usage") or {}
        diagnostics = payload.get("diagnostics") or {}
        calls = diagnostics.get("model_calls") if role != "reviewer" else payload.get("attempts")
        return cls(
            role=role, payload=payload, model_calls=max(0, int(calls if calls is not None else 1)),
            input_tokens=max(0, int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)),
            output_tokens=max(0, int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)),
            elapsed_seconds=max(0.0, elapsed_seconds),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "role": self.role, "model_calls": self.model_calls,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "answer": self.payload.get("answer") if self.role != "reviewer" else self.payload.get("proposed_answer"),
            "verdict": self.payload.get("verdict") if self.role == "reviewer" else None,
            "error_type": self.payload.get("error_type"),
        }


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class TaskBudget:
    """A shared cap across preparation, solver, reviewer, and adjudicator."""

    max_model_calls: int = 90
    max_input_tokens: int = 360_000
    max_output_tokens: int = 75_000
    max_seconds: float = 1_200
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    prior_elapsed_seconds: float = 0
    started_at: float = field(default_factory=time.monotonic, repr=False)

    @classmethod
    def from_previous(cls, previous: dict[str, Any]) -> TaskBudget:
        saved = previous.get("budget")
        if isinstance(saved, dict):
            return cls(
                model_calls=max(0, int(saved.get("model_calls", 0))),
                input_tokens=max(0, int(saved.get("input_tokens", 0))),
                output_tokens=max(0, int(saved.get("output_tokens", 0))),
                prior_elapsed_seconds=max(0.0, float(saved.get("elapsed_seconds", 0))),
            )
        budget = cls()
        for role, key in (("solver", "primary"), ("reviewer", "review"), ("judge", "final")):
            payload = previous.get(key)
            if not isinstance(payload, dict):
                continue
            if role == "judge" and (payload == previous.get("primary") or not previous.get("review")):
                continue
            stage = StageResult.from_payload(role, payload)
            budget.model_calls += stage.model_calls
            budget.input_tokens += stage.input_tokens
            budget.output_tokens += stage.output_tokens
        return budget

    @property
    def elapsed_seconds(self) -> float:
        return self.prior_elapsed_seconds + time.monotonic() - self.started_at

    @property
    def remaining_model_calls(self) -> int:
        return max(0, self.max_model_calls - self.model_calls)

    def ensure(self, role: str, required_calls: int = 1) -> None:
        if self.remaining_model_calls < required_calls:
            raise BudgetExceeded(f"{role}: task model-call budget exhausted")
        if self.input_tokens >= self.max_input_tokens or self.output_tokens >= self.max_output_tokens:
            raise BudgetExceeded(f"{role}: task token budget exhausted")
        if self.elapsed_seconds >= self.max_seconds:
            raise BudgetExceeded(f"{role}: task wall-time budget exhausted")

    def charge(self, stage: StageResult) -> None:
        self.model_calls += stage.model_calls
        self.input_tokens += stage.input_tokens
        self.output_tokens += stage.output_tokens
        if self.model_calls > self.max_model_calls or self.input_tokens > self.max_input_tokens or self.output_tokens > self.max_output_tokens:
            raise BudgetExceeded(f"{stage.role}: task budget exceeded")
        if self.elapsed_seconds >= self.max_seconds:
            raise BudgetExceeded(f"{stage.role}: task wall-time budget exhausted")

    def exhausted(self) -> bool:
        return (self.remaining_model_calls == 0 or self.input_tokens >= self.max_input_tokens
                or self.output_tokens >= self.max_output_tokens or self.elapsed_seconds >= self.max_seconds)

    def to_record(self) -> dict[str, Any]:
        return {
            "model_calls": self.model_calls, "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens, "elapsed_seconds": round(self.elapsed_seconds, 3),
            "limits": {"model_calls": self.max_model_calls, "input_tokens": self.max_input_tokens,
                       "output_tokens": self.max_output_tokens, "seconds": self.max_seconds},
        }


@dataclass(frozen=True)
class SkillDescriptor:
    name: str
    version: str


@dataclass
class TaskContext:
    task: Task
    prepared: PreparedInput
    memory: TaskMemory
    budget: TaskBudget
    skills: tuple[SkillDescriptor, ...] = ()


class AgentRole(Protocol):
    name: str

    def run(self, context: TaskContext, primary: dict[str, Any] | None = None,
            review: dict[str, Any] | None = None) -> StageResult: ...
