from dataclasses import replace
from types import SimpleNamespace

import pytest

from gaia_agent.agent import GaiaAgent, PreparedInput, _degraded_review
from gaia_agent.config import load_config
from gaia_agent.contracts import StageResult, TaskBudget
from gaia_agent.memory import TaskMemory
from gaia_agent.reviewer import _review_content
from gaia_agent.sources import Task


def _agent(monkeypatch, *, mode="always"):
    agent = GaiaAgent.__new__(GaiaAgent)
    agent.config = SimpleNamespace(
        solver_provider="deepseek", solver_model="solver",
        judge_provider="deepseek", judge_model="judge",
        review_mode=mode, confidence_threshold=0.8,
        code_interpreter_enabled=False,
    )
    monkeypatch.setattr(agent, "_prepare", lambda task: PreparedInput(content=[{"type": "input_text", "text": task.question}]))
    return agent


def test_review_off_needs_only_solver_provider_key():
    config = replace(
        load_config("config.example.toml"), review_mode="off",
        deepseek_key="test-solver-key", judge_provider="openai", openai_key="",
    )
    agent = GaiaAgent(config)
    assert hasattr(agent, "deepseek")
    assert not hasattr(agent, "openai")


def test_enabled_review_requires_reviewer_key_before_batch_starts():
    config = replace(
        load_config("config.example.toml"), review_mode="adaptive",
        deepseek_key="test-solver-key", anthropic_key="",
    )
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        GaiaAgent(config)


def test_roles_share_budget_and_checkpoint_typed_memory(monkeypatch):
    agent = _agent(monkeypatch)
    remaining = []

    def ask(**kwargs):
        remaining.append(kwargs["remaining_model_calls"])
        role = kwargs["model"]
        calls = 79 if role == "solver" else 5
        return {
            "answer": "A" if role == "solver" else "B",
            "confidence": 0.9, "evidence": [],
            "diagnostics": {"model_calls": calls},
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

    monkeypatch.setattr(agent, "_ask_model", ask)
    monkeypatch.setattr(agent, "_review", lambda *args: {
        "verdict": "disagree", "proposed_answer": "B", "attempts": 5,
        "usage": {"input_tokens": 4, "output_tokens": 2},
    })
    checkpoints = []
    result = agent.solve(Task("abc", "Question?", 2), checkpoint=checkpoints.append)
    assert remaining == [90, 6]
    assert result["status"] == "completed"
    assert result["budget"]["model_calls"] == 89
    assert [stage["role"] for stage in result["task_memory"]["stages"]] == ["solver", "reviewer", "judge"]
    assert [item["status"] for item in checkpoints] == ["primary_completed", "review_completed"]
    assert checkpoints[0]["budget"]["model_calls"] == 79
    assert checkpoints[1]["budget"]["model_calls"] == 84


def test_exhausted_shared_budget_never_completes(monkeypatch):
    agent = _agent(monkeypatch)

    def ask(**kwargs):
        calls = 80 if kwargs["model"] == "solver" else 5
        return {"answer": kwargs["model"], "confidence": 0.9, "evidence": [],
                "diagnostics": {"model_calls": calls}}

    monkeypatch.setattr(agent, "_ask_model", ask)
    monkeypatch.setattr(agent, "_review", lambda *args: {
        "verdict": "disagree", "proposed_answer": "different", "attempts": 5,
    })
    result = agent.solve(Task("abc", "Question?", 2))
    assert result["budget"]["model_calls"] == 90
    assert result["status"] == "budget_exhausted"


def test_remote_transcription_consumes_shared_budget(monkeypatch):
    agent = _agent(monkeypatch, mode="off")
    agent.config.transcription_provider = "openai"
    monkeypatch.setattr(agent, "_prepare", lambda task: PreparedInput(
        content=[{"type": "input_text", "text": task.question}], transcript="spoken words",
    ))
    remaining = []
    monkeypatch.setattr(agent, "_ask_model", lambda **kwargs: remaining.append(kwargs["remaining_model_calls"]) or {
        "answer": "42", "confidence": 0.9, "evidence": [], "diagnostics": {"model_calls": 1},
    })
    result = agent.solve(Task("abc", "Audio?", 1))
    assert remaining == [89]
    assert result["budget"]["model_calls"] == 2
    assert [stage["role"] for stage in result["task_memory"]["stages"]] == ["preparation", "solver"]


def test_task_memory_does_not_follow_another_task_and_links_visual_source(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"image")
    memory = TaskMemory("task-a")
    memory.add_prepared(PreparedInput(
        content=[], review_images=[(frame, 12.5, "Video frame near 12.5 seconds")],
    ))
    memory.add_model_result(StageResult.from_payload("solver", {
        "answer": "42", "sources": [{"url": "https://example.org/source", "status": "fetched", "excerpt": "fact"}],
        "vision_outputs": [{"image_index": 0, "label": "Video frame near 12.5 seconds", "observation": "red sign",
                            "question": "What color is the sign?", "region": [0, 0, 50, 50]}],
        "tool_trace": [{"tool": "query_attachment", "source_path": "sheet.xlsx", "sha256": "abc",
                        "result_preview": [{"total": 42}], "truncated": False}],
    }))
    assert memory.observations[1].asset_id == memory.observations[0].asset_id
    assert memory.observations[1].timestamp == 12.5
    assert memory.observations[1].text == "red sign"
    assert memory.observations[1].question == "What color is the sign?"
    assert memory.observations[1].region == "[0, 0, 50, 50]"
    assert memory.evidence[0].locator == "https://example.org/source"
    assert any(item.kind == "data_query" and item.sha256 == "abc" for item in memory.evidence)
    assert TaskMemory.from_previous("task-b", {"task_memory": memory.to_record()}).evidence == []


def test_reviewer_prompt_is_blind_to_primary_answer_and_source_selection():
    task = Task("abc", "Which sign?", 1)
    primary = {"answer": "secret answer", "sources": [
        {"status": "fetched", "url": "https://example.org/biased", "excerpt": "secret answer"},
    ]}
    content = _review_content(task, primary, PreparedInput(content=[], review_text="Attachment: red sign"))
    prompt = content[0]["text"]
    assert "Which sign?" in prompt
    assert "Attachment: red sign" in prompt
    assert "secret answer" not in prompt
    assert "https://example.org/biased" not in prompt


def test_legacy_checkpoint_budget_is_reconstructed():
    previous = {"primary": {"answer": "42", "usage": {"input_tokens": 10, "output_tokens": 2},
                            "diagnostics": {"model_calls": 3}}}
    budget = TaskBudget.from_previous(previous)
    assert budget.model_calls == 3
    assert budget.input_tokens == 10
    assert budget.output_tokens == 2


def test_resume_checkpoint_preserves_review_during_judge_retry(monkeypatch):
    agent = _agent(monkeypatch)
    monkeypatch.setattr(agent, "_ask_model", lambda **kwargs: {
        "answer": "B", "confidence": 0.8, "evidence": [],
        "diagnostics": {"model_calls": 1},
    })
    monkeypatch.setattr(agent, "_review", lambda *args: pytest.fail("saved review must be reused"))
    review = {"verdict": "disagree", "proposed_answer": "B", "attempts": 1}
    previous = {
        "status": "review_completed", "primary": {"answer": "A", "confidence": 0.9, "evidence": []},
        "review": review,
    }
    checkpoints = []
    result = agent.solve(Task("abc", "Question?", 2), previous=previous, checkpoint=checkpoints.append)
    assert result["status"] == "completed"
    assert len(checkpoints) == 1
    assert checkpoints[0]["status"] == "review_completed"
    assert checkpoints[0]["review"] == review


def test_partial_local_tool_result_triggers_review_and_export_warning(monkeypatch):
    agent = _agent(monkeypatch, mode="adaptive")
    monkeypatch.setattr(agent, "_ask_model", lambda **kwargs: {
        "answer": "42", "confidence": 0.99, "evidence": [],
        "tool_trace": [{"tool": "query_attachment", "truncated": True,
                        "warning": "Only the first 200 rows were returned"}],
    })
    reviewed = []
    monkeypatch.setattr(agent, "_review", lambda *args: reviewed.append(True) or {
        "verdict": "agree", "proposed_answer": "42", "attempts": 1,
    })
    result = agent.solve(Task("abc", "Compute a total?", 1))
    assert reviewed == [True]
    assert result["status"] == "completed"
    assert result["warnings"] == [
        "query_attachment returned partial evidence: Only the first 200 rows were returned"
    ]


def test_failed_local_tool_result_is_not_silent(monkeypatch):
    agent = _agent(monkeypatch, mode="off")
    monkeypatch.setattr(agent, "_ask_model", lambda **kwargs: {
        "answer": "42", "confidence": 0.9, "evidence": [],
        "tool_trace": [{"tool": "calculate", "error": "invalid expression"}],
    })
    result = agent.solve(Task("abc", "Question?", 1))
    assert result["warnings"] == ["calculate failed: invalid expression"]


def test_explicitly_sound_no_tools_review_is_not_rejected_as_legacy():
    assert not _degraded_review({"review_mode": "no_tools_structured", "evidence_degraded": False})
    assert _degraded_review({"review_mode": "no_tools_structured"})


def test_failed_judge_calls_are_charged_to_shared_budget(monkeypatch):
    agent = _agent(monkeypatch)

    class JudgeFailure(RuntimeError):
        diagnostics = {"model_calls": 3}
        usage = {"input_tokens": 120, "output_tokens": 20}

    def ask(**kwargs):
        if kwargs["model"] == "judge":
            raise JudgeFailure("upstream failed")
        return {"answer": "A", "confidence": 0.9, "evidence": [],
                "diagnostics": {"model_calls": 1}}

    monkeypatch.setattr(agent, "_ask_model", ask)
    monkeypatch.setattr(agent, "_review", lambda *args: {
        "verdict": "disagree", "proposed_answer": "B", "attempts": 1,
    })
    result = agent.solve(Task("abc", "Question?", 2))
    assert result["status"] == "needs_adjudication"
    assert result["budget"]["model_calls"] == 5
    assert result["budget"]["input_tokens"] == 120
    assert result["task_memory"]["stages"][-1]["error_type"] == "JudgeFailure"
