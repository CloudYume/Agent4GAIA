from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from gaia_agent.agent import GaiaAgent, PreparedInput, _openai_tools, _safe_error_detail, _solver_prompt, clean_answer, is_quota_error
from gaia_agent.config import load_config
from gaia_agent.sources import Task


def test_clean_answer():
    assert clean_answer(" FINAL ANSWER: 42 ") == "42"
    with pytest.raises(ValueError, match="one nonempty, short line"):
        clean_answer("first\rsecond")


def test_quota_error_does_not_confuse_rate_limit():
    quota = type("APIError", (Exception,), {"status_code": 402, "code": "insufficient_balance"})()
    rate_limit = type("APIError", (Exception,), {"status_code": 429, "code": "rate_limit"})()
    assert is_quota_error(quota)
    assert not is_quota_error(rate_limit)


def test_code_interpreter_toggle_changes_tools_and_solver_prompt():
    enabled = replace(load_config("config.example.toml"), code_interpreter_enabled=True)
    disabled = replace(enabled, code_interpreter_enabled=False)
    assert [tool["type"] for tool in _openai_tools(enabled)] == ["web_search", "code_interpreter"]
    assert [tool["type"] for tool in _openai_tools(disabled)] == ["web_search"]
    assert _openai_tools(enabled)[0]["search_context_size"] == "medium"
    assert "Python code interpreter" in _solver_prompt(enabled)
    assert "Python" not in _solver_prompt(disabled)


def test_safe_error_detail_extracts_failure_and_redacts_tokens():
    event = type("Event", (), {
        "type": "response.failed",
        "response": type("Response", (), {
            "error": type("Error", (), {
                "code": "server_error",
                "message": "upstream https://example.invalid?api_key=sk-secret-token-value",
                "param": "input",
            })(),
        })(),
    })()
    detail = _safe_error_detail(event)
    assert "code=server_error" in detail
    assert "param=input" in detail
    assert "[REDACTED]" in detail
    assert "sk-secret-token-value" not in detail


@pytest.mark.parametrize("max_tool_calls", [0, 3])
def test_stream_failure_includes_error_event_detail(max_tool_calls):
    agent = GaiaAgent.__new__(GaiaAgent)
    agent.config = type("Config", (), {
        "reasoning_effort": "high", "code_interpreter_enabled": False,
        "search_context_size": "low", "max_tool_calls": max_tool_calls, "max_output_tokens": 1200,
    })()
    request = {}

    class Responses:
        def create(self, **kwargs):
            request.update(kwargs)
            return iter([type("Event", (), {
                "type": "response.failed",
                "response": type("Response", (), {
                    "error": type("Error", (), {"code": "invalid_prompt", "message": "bad input", "param": "input"})(),
                })(),
            })()])

    agent.openai = type("Client", (), {"responses": Responses()})()
    with pytest.raises(RuntimeError, match="invalid_prompt.*bad input.*param=input"):
        agent._ask_openai(model="solver", instructions="prompt", content=[])
    assert [tool["type"] for tool in request["tools"]] == ["web_search"]
    assert request["tools"][0]["search_context_size"] == "low"
    if max_tool_calls:
        assert request["max_tool_calls"] == 3
    else:
        assert "max_tool_calls" not in request
    assert request["max_output_tokens"] == 1200


def test_completed_response_records_tool_counts_and_request_budget():
    agent = GaiaAgent.__new__(GaiaAgent)
    agent.config = replace(
        load_config("config.example.toml"), search_context_size="high",
        max_tool_calls=5, max_output_tokens=2500,
    )
    request = {}
    response = SimpleNamespace(
        status="completed", id="resp-test", output_text='{"answer":"42","confidence":0.9,"evidence":[]}',
        usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 100, "output_tokens": 20}),
        output=[
            SimpleNamespace(type="web_search_call"),
            SimpleNamespace(type="web_search_call"),
            SimpleNamespace(type="code_interpreter_call"),
            SimpleNamespace(type="message", content=[SimpleNamespace(annotations=[])]),
        ],
    )

    class Responses:
        def create(self, **kwargs):
            request.update(kwargs)
            return iter([SimpleNamespace(type="response.completed", response=response)])

    agent.openai = SimpleNamespace(responses=Responses())
    result = agent._ask_openai(model="solver", instructions="prompt", content=[])
    assert request["tools"][0]["search_context_size"] == "high"
    assert request["max_tool_calls"] == 5
    assert request["max_output_tokens"] == 2500
    assert result["usage"] == {"input_tokens": 100, "output_tokens": 20}
    assert result["tool_counts"] == {
        "web_search_call": 2, "code_interpreter_call": 1, "total_tool_items": 3,
    }


def test_safe_error_detail_accepts_dict_error_event():
    detail = _safe_error_detail({
        "type": "error",
        "error": {"type": "rate_limit_error", "code": "rate_limit", "message": 'Authorization: Bearer custom-token; "api_key":"private-value"'},
    })
    assert "code=rate_limit" in detail
    assert "type=rate_limit_error" in detail
    assert "custom-token" not in detail
    assert "private-value" not in detail


def test_disagreement_invokes_judge(monkeypatch):
    agent = GaiaAgent.__new__(GaiaAgent)
    agent.config = type("Config", (), {"review_mode": "always", "confidence_threshold": 0.8, "code_interpreter_enabled": True})()
    agent.openai = type("Client", (), {"files": type("Files", (), {"delete": lambda self, _: None})()})()
    monkeypatch.setattr(agent, "_prepare", lambda task: PreparedInput(content=[{"type": "input_text", "text": task.question}]))
    calls = []

    def ask(**kwargs):
        calls.append(kwargs["model"])
        answer = "A" if len(calls) == 1 else "B"
        return {"answer": answer, "confidence": 0.9, "evidence": [], "model": kwargs["model"]}

    monkeypatch.setattr(agent, "_ask_openai", ask)
    monkeypatch.setattr(
        agent,
        "_review",
        lambda task, primary, prepared: {"verdict": "disagree", "proposed_answer": "B", "reason": "source"},
    )
    agent.config.solver_model = "solver"
    agent.config.judge_model = "judge"
    result = agent.solve(Task("abc", "Question?", 1))
    assert calls == ["solver", "judge"]
    assert result["answer"] == "B"


def test_equivalent_review_answer_does_not_invoke_judge(monkeypatch):
    agent = GaiaAgent.__new__(GaiaAgent)
    agent.config = type("Config", (), {
        "review_mode": "always", "confidence_threshold": 0.8,
        "solver_model": "solver", "judge_model": "judge", "code_interpreter_enabled": False,
    })()
    monkeypatch.setattr(agent, "_prepare", lambda task: PreparedInput(content=[{"type": "input_text", "text": task.question}]))
    calls = []

    def ask(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "judge":
            pytest.fail("equivalent reviewer answer must not trigger judge")
        return {"answer": "42", "confidence": 0.9, "evidence": [], "model": "solver"}

    monkeypatch.setattr(agent, "_ask_openai", ask)
    monkeypatch.setattr(agent, "_review", lambda *args: {
        "verdict": "agree", "proposed_answer": "42.0", "reason": "same numeric answer", "evidence": [],
        "evidence_degraded": False,
    })
    result = agent.solve(Task("abc", "Question?", 1))
    assert result["status"] == "completed"
    assert result["answer"] == "42"
    assert calls == ["solver"]


def test_review_failure_preserves_primary_answer(monkeypatch):
    agent = GaiaAgent.__new__(GaiaAgent)
    agent.config = type("Config", (), {
        "review_mode": "always", "confidence_threshold": 0.8,
        "code_interpreter_enabled": True,
        "solver_model": "solver", "judge_model": "judge",
    })()
    monkeypatch.setattr(agent, "_prepare", lambda task: PreparedInput(content=[{"type": "input_text", "text": task.question}]))
    monkeypatch.setattr(agent, "_ask_openai", lambda **kwargs: {
        "answer": "42", "confidence": 0.9, "evidence": ["source"], "model": kwargs["model"],
    })

    def unavailable(*args):
        raise TimeoutError("review unavailable")

    monkeypatch.setattr(agent, "_review", unavailable)
    result = agent.solve(Task("abc", "Question?", 2))
    assert result["status"] == "needs_review"
    assert result["answer"] == "42"
    assert result["review_error"] == "TimeoutError: review unavailable"


def test_review_retry_reuses_saved_primary(monkeypatch):
    agent = GaiaAgent.__new__(GaiaAgent)
    agent.config = type("Config", (), {
        "review_mode": "always", "confidence_threshold": 0.8,
        "solver_model": "solver", "judge_model": "judge", "code_interpreter_enabled": False,
    })()
    monkeypatch.setattr(agent, "_prepare", lambda task: PreparedInput(content=[{"type": "input_text", "text": task.question}]))
    solver_calls = []
    monkeypatch.setattr(agent, "_ask_openai", lambda **kwargs: solver_calls.append(kwargs) or {
        "answer": "42", "confidence": 0.9, "evidence": [], "model": "solver",
    })
    review_calls = []

    def review(*args):
        review_calls.append(True)
        if len(review_calls) == 1:
            raise TimeoutError("review unavailable")
        return {"verdict": "agree", "proposed_answer": "42", "reason": "checked", "evidence": []}

    monkeypatch.setattr(agent, "_review", review)
    saved = []
    task = Task("abc", "Question?", 1)
    first = agent.solve(task, checkpoint=saved.append)
    second = agent.solve(task, previous=first, checkpoint=saved.append)
    assert first["status"] == "needs_review"
    assert second["status"] == "completed"
    assert len(solver_calls) == 1
    assert len(review_calls) == 2
    assert [row["status"] for row in saved] == ["primary_completed", "primary_completed", "review_completed"]


def test_degraded_agreement_needs_review_and_does_not_reuse_it(monkeypatch):
    agent = GaiaAgent.__new__(GaiaAgent)
    agent.config = type("Config", (), {
        "review_mode": "always", "confidence_threshold": 0.8,
        "solver_model": "solver", "judge_model": "judge", "code_interpreter_enabled": False,
    })()
    monkeypatch.setattr(agent, "_prepare", lambda task: PreparedInput(content=[{"type": "input_text", "text": task.question}]))
    solver_calls = []
    monkeypatch.setattr(agent, "_ask_openai", lambda **kwargs: solver_calls.append(kwargs["model"]) or {
        "answer": "42", "confidence": 0.9, "evidence": [], "model": kwargs["model"],
    })
    review_calls = []

    def review(*args):
        review_calls.append(True)
        return {
            "verdict": "agree", "proposed_answer": "42", "reason": "checked", "evidence": [],
            "evidence_degraded": len(review_calls) == 1,
            "degradation_reason": "web_search_finalization_failed" if len(review_calls) == 1 else "",
        }

    monkeypatch.setattr(agent, "_review", review)
    saved = []
    task = Task("abc", "Question?", 1)
    first = agent.solve(task, checkpoint=saved.append)
    second = agent.solve(task, previous=first, checkpoint=saved.append)
    assert first["status"] == "needs_review"
    assert first["review_error"] == "Reviewer evidence degraded: web_search_finalization_failed"
    assert first["review"]["verdict"] == "agree"
    assert second["status"] == "completed"
    assert solver_calls == ["solver"]
    assert len(review_calls) == 2
    assert [row["status"] for row in saved] == ["primary_completed", "primary_completed", "review_completed"]


def test_resume_rechecks_legacy_no_tools_review_without_degradation_flag(monkeypatch):
    agent = GaiaAgent.__new__(GaiaAgent)
    agent.config = type("Config", (), {
        "review_mode": "always", "confidence_threshold": 0.8,
        "solver_model": "solver", "judge_model": "judge", "code_interpreter_enabled": False,
    })()
    monkeypatch.setattr(agent, "_prepare", lambda task: PreparedInput(content=[{"type": "input_text", "text": task.question}]))
    monkeypatch.setattr(agent, "_ask_openai", lambda **kwargs: pytest.fail("primary must be reused"))
    reviews = []
    monkeypatch.setattr(agent, "_review", lambda *args: reviews.append(True) or {
        "verdict": "agree", "proposed_answer": "42", "reason": "checked", "evidence": [],
        "review_mode": "web_search", "evidence_degraded": False,
    })
    previous = {
        "primary": {"answer": "42", "confidence": 0.9, "evidence": []},
        "review": {"verdict": "agree", "proposed_answer": "42", "review_mode": "no_tools_structured"},
    }
    result = agent.solve(Task("abc", "Question?", 1), previous=previous)
    assert result["status"] == "completed"
    assert reviews == [True]


def test_video_evidence_triggers_adaptive_review(monkeypatch):
    agent = GaiaAgent.__new__(GaiaAgent)
    agent.config = type("Config", (), {
        "review_mode": "adaptive", "confidence_threshold": 0.8,
        "code_interpreter_enabled": True,
        "solver_model": "solver", "judge_model": "judge",
    })()
    monkeypatch.setattr(agent, "_prepare", lambda task: PreparedInput(
        content=[{"type": "input_text", "text": task.question}],
        review_images=[(Path("frame.jpg"), 12.0, "Video frame near 12.0 seconds")],
    ))
    monkeypatch.setattr(agent, "_ask_openai", lambda **kwargs: {
        "answer": "42", "confidence": 0.99, "evidence": [], "model": kwargs["model"],
    })
    reviewed = []
    monkeypatch.setattr(agent, "_review", lambda *args: reviewed.append(True) or {
        "verdict": "agree", "proposed_answer": "42", "reason": "verified", "evidence": [],
    })
    result = agent.solve(Task("abc", "Question with a video?", 1))
    assert reviewed == [True]
    assert result["answer"] == "42"


def test_failed_adjudication_is_not_completed(monkeypatch):
    agent = GaiaAgent.__new__(GaiaAgent)
    agent.config = type("Config", (), {
        "review_mode": "always", "confidence_threshold": 0.8,
        "code_interpreter_enabled": True,
        "solver_model": "solver", "judge_model": "judge",
    })()
    monkeypatch.setattr(agent, "_prepare", lambda task: PreparedInput(content=[{"type": "input_text", "text": task.question}]))

    def ask(**kwargs):
        if kwargs["model"] == "judge":
            raise TimeoutError("judge unavailable")
        return {"answer": "A", "confidence": 0.9, "evidence": [], "model": "solver"}

    monkeypatch.setattr(agent, "_ask_openai", ask)
    monkeypatch.setattr(agent, "_review", lambda *args: {
        "verdict": "disagree", "proposed_answer": "B", "reason": "different source", "evidence": [],
    })
    result = agent.solve(Task("abc", "Question?", 2))
    assert result["status"] == "needs_adjudication"
    assert result["answer"] == "A"
    assert result["review"]["proposed_answer"] == "B"
    assert result["adjudication_error"] == "TimeoutError: judge unavailable"
