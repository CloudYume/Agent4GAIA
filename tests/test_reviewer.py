from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from gaia_agent import reviewer
from gaia_agent.attachments import PreparedInput
from gaia_agent.sources import Task


def _message(content, *, stop_reason="end_turn", response_id="review", input_tokens=10, output_tokens=5):
    return {
        "id": response_id, "content": content, "stop_reason": stop_reason,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _answer(proposed_answer="42", certainty="supported"):
    return {
        "type": "text",
        "text": '{"proposed_answer":"' + proposed_answer + '","certainty":"' + certainty + '","reason":"checked","evidence":["source"]}',
    }


def _client(monkeypatch, responses):
    calls = []
    queue = iter(responses)

    def create(**kwargs):
        calls.append(deepcopy(kwargs))
        response = next(queue)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(reviewer, "Anthropic", lambda **kwargs: SimpleNamespace(messages=SimpleNamespace(create=create)))
    return calls


def _config():
    return SimpleNamespace(
        reviewer_provider="anthropic", reviewer_model="claude-reviewer",
        anthropic_key="sk-test", anthropic_base_url="", require=lambda provider: None,
    )


def _run(prepared=None):
    return reviewer.review_anthropic(
        _config(), Task("abc", "Question?", 1),
        {"answer": "42", "evidence": ["source"]}, prepared or PreparedInput(content=[]),
    )


def test_review_selects_final_json_block_and_records_sources(monkeypatch):
    calls = _client(monkeypatch, [_message([
        {"type": "server_tool_use", "id": "search"},
        {"type": "web_search_tool_result", "content": [{"url": "https://example.com/source"}]},
        {"type": "text", "text": ""},
        _answer("43"),
    ])])
    result = _run()
    assert result["verdict"] == "disagree"
    assert result["citations"] == ["https://example.com/source"]
    assert result["usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert result["attempts"] == 1
    assert result["review_mode"] == "web_search"
    assert result["evidence_degraded"] is False
    assert calls[0]["tools"][0]["name"] == "web_search"
    assert "Primary answer" not in calls[0]["messages"][0]["content"][0]["text"]
    assert "42" not in calls[0]["messages"][0]["content"][0]["text"]


def test_review_continues_pause_turn_with_server_tool_blocks(monkeypatch):
    calls = _client(monkeypatch, [
        _message([{"type": "web_search_tool_result", "content": [{"url": "https://example.com/"}]}], stop_reason="pause_turn"),
        _message([_answer()], response_id="finished"),
    ])
    result = _run()
    assert result["response_id"] == "finished"
    assert result["attempts"] == 2
    assert result["usage"] == {"input_tokens": 20, "output_tokens": 10}
    assert calls[1]["messages"][1]["role"] == "assistant"
    assert calls[1]["messages"][1]["content"][0]["type"] == "web_search_tool_result"


def test_review_retries_empty_tool_response_without_tools(monkeypatch):
    calls = _client(monkeypatch, [
        _message([{"type": "server_tool_use", "id": "search"}]),
        _message([_answer()]),
    ])
    result = _run()
    assert result["review_mode"] == "no_tools_structured"
    assert result["evidence_degraded"] is True
    assert result["attempts"] == 2
    assert "tools" in calls[0]
    assert "tools" not in calls[1]
    assert "output_config" in calls[1]


def test_review_uses_plain_json_retry_if_structured_response_is_empty(monkeypatch):
    calls = _client(monkeypatch, [
        _message([], stop_reason="end_turn"),
        _message([{"type": "text", "text": ""}]),
        _message([_answer()]),
    ])
    result = _run()
    assert result["review_mode"] == "no_tools_plain_json"
    assert result["evidence_degraded"] is True
    assert result["attempts"] == 3
    assert "tools" not in calls[2]
    assert "output_config" not in calls[2]


def test_review_retries_max_tokens_even_if_json_text_looks_complete(monkeypatch):
    calls = _client(monkeypatch, [
        _message([_answer("43")], stop_reason="max_tokens"),
        _message([_answer("42")]),
    ])
    result = _run()
    assert result["verdict"] == "agree"
    assert result["review_mode"] == "no_tools_structured"
    assert result["evidence_degraded"] is True
    assert len(calls) == 2


def test_review_replays_web_counterevidence_during_no_tools_fallback(monkeypatch):
    calls = _client(monkeypatch, [
        _message([{
            "type": "web_search_tool_result",
            "content": [{"url": "https://example.com/counter", "snippet": "Counterexample: 43"}],
        }]),
        _message([_answer("43")]),
    ])
    result = _run()
    fallback_text = calls[1]["messages"][0]["content"][-1]["text"]
    assert "Counterexample: 43" in fallback_text
    assert result["verdict"] == "disagree"
    assert result["evidence_degraded"] is True
    assert result["citations"] == ["https://example.com/counter"]


def test_review_programmatically_compares_equivalent_numeric_answers(monkeypatch):
    _client(monkeypatch, [_message([_answer("42.0")])])
    result = _run()
    assert result["verdict"] == "agree"
    assert result["proposed_answer"] == "42.0"


def test_review_preserves_uncertain_blind_answer(monkeypatch):
    _client(monkeypatch, [_message([_answer("", "uncertain")])])
    assert _run()["verdict"] == "uncertain"


def test_review_preserves_quota_status_without_secret(monkeypatch):
    class QuotaError(Exception):
        status_code = 429
        body = {"error": {"code": "insufficient_quota", "message": "sk-private-key"}}

    calls = _client(monkeypatch, [QuotaError("sk-private-key")])
    with pytest.raises(reviewer.ReviewerAPIError, match="status=429") as error:
        _run()
    assert "quota_or_rate_limit" in str(error.value)
    assert error.value.attempts == 1
    assert "sk-private-key" not in str(error.value)
    assert len(calls) == 1


def test_review_reports_structural_failure_without_echoing_text(monkeypatch):
    _client(monkeypatch, [_message([{"type": "text", "text": "sk-private-key"}]) for _ in range(3)])
    with pytest.raises(reviewer.ReviewerFormatError, match="no valid JSON answer") as error:
        _run()
    assert "sk-private-key" not in str(error.value)


def test_review_forwards_labeled_images(monkeypatch, tmp_path):
    calls = _client(monkeypatch, [_message([_answer()])])
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"image")
    prepared = PreparedInput(content=[], review_images=[(Path(frame), 12.0, "Video frame near 12 seconds")])
    _run(prepared)
    content = calls[0]["messages"][0]["content"]
    assert content[1]["text"] == "Video frame near 12 seconds"
    assert content[2]["source"]["media_type"] == "image/jpeg"
    assert content[2]["source"]["data"] == "aW1hZ2U="


def _local_review(monkeypatch, responses, primary=None):
    calls = _client(monkeypatch, responses)
    result = reviewer.review_anthropic(
        _config(), Task("abc", "Read the attached note", 1, "note.txt", "note.txt"),
        primary or {"answer": "42"},
        PreparedInput(content=[], review_text="Note contents: 42"),
    )
    return result, calls


def test_attachment_only_no_tools_format_repair_is_usable(monkeypatch):
    result, calls = _local_review(monkeypatch, [
        _message([{"type": "text", "text": ""}]),
        _message([_answer()]),
    ])
    assert len(calls) == 2
    assert result["review_mode"] == "no_tools_structured"
    assert result["fallback_reason"] == "incomplete_structured_response"
    assert result["evidence_degraded"] is False
    assert result["verdict"] == "agree"


def test_no_tools_repair_cannot_launder_primary_web_dependence(monkeypatch):
    result, _ = _local_review(monkeypatch, [
        _message([{"type": "text", "text": ""}]),
        _message([_answer()]),
    ], primary={"answer": "42", "sources": [{"url": "https://example.org", "status": "fetched"}]})
    assert result["web_evidence_required"] is True
    assert result["evidence_degraded"] is True


def test_no_tools_repair_cannot_launder_lost_web_tool_result(monkeypatch):
    result, _ = _local_review(monkeypatch, [
        _message([{"type": "server_tool_use", "id": "search"}]),
        _message([_answer()]),
    ])
    assert result["web_tool_attempted"] is True
    assert result["evidence_degraded"] is True
    assert result["degradation_reason"] == "web_search_result_incomplete"


def test_unsupported_server_tool_can_repair_local_only_question(monkeypatch):
    class UnsupportedTool(Exception):
        status_code = 400
        body = {"error": {"code": "invalid_request_error", "param": "tools.0",
                          "message": "web_search unsupported"}}

    result, calls = _local_review(monkeypatch, [UnsupportedTool(), _message([_answer()])])
    assert len(calls) == 2
    assert result["fallback_reason"] == "unsupported_server_web_search"
    assert result["evidence_degraded"] is False
    assert result["attempts"] == 2


def test_unsupported_server_tool_stays_fatal_without_local_evidence(monkeypatch):
    class UnsupportedTool(Exception):
        status_code = 400
        body = {"error": {"code": "invalid_request_error", "param": "tools.0"}}

    _client(monkeypatch, [UnsupportedTool()])
    with pytest.raises(reviewer.ReviewerAPIError) as error:
        _run()
    assert error.value.unsupported_tool is True


def test_quota_error_is_never_reclassified_as_unsupported_tool(monkeypatch):
    class QuotaError(Exception):
        status_code = 400
        body = {"error": {"code": "insufficient_quota", "param": "tools.0"}}

    _client(monkeypatch, [QuotaError()])
    with pytest.raises(reviewer.ReviewerAPIError) as error:
        reviewer.review_anthropic(
            _config(), Task("abc", "Read the attached note", 1, "note.txt", "note.txt"),
            {"answer": "42"}, PreparedInput(content=[], review_text="Note contents: 42"),
        )
    assert error.value.unsupported_tool is False
    assert error.value.code == "insufficient_quota"
