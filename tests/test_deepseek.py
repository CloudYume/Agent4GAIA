import base64
import io
from copy import deepcopy
from types import SimpleNamespace

import pytest
import requests
from PIL import Image

from gaia_agent import deepseek, web_tools


def _response(message, *, response_id="resp", finish_reason="stop", prompt_tokens=10, completion_tokens=5):
    return SimpleNamespace(
        id=response_id,
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, prompt_cache_hit_tokens=2),
    )


def _client(responses):
    calls = []
    queue = iter(responses)

    def create(**kwargs):
        calls.append(deepcopy(kwargs))
        result = next(queue)
        if isinstance(result, Exception):
            raise result
        return result

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls


def _config(max_tool_calls=3):
    return SimpleNamespace(max_tool_calls=max_tool_calls, max_output_tokens=1000, reasoning_effort="high", tavily_api_key="")


def _tool_call(call_id, name, arguments):
    return {"id": call_id, "function": {"name": name, "arguments": arguments}}


def _image_url():
    output = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(output, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def test_deepseek_search_fetch_loop_records_sources_and_usage(monkeypatch):
    monkeypatch.setattr(deepseek, "search_web", lambda query, **kwargs: {
        "provider": "bing_rss", "query": query,
        "results": [{"title": "Source", "url": "https://example.com/source", "snippet": "Evidence"}],
    })
    monkeypatch.setattr(deepseek, "fetch_url", lambda url: {
        "url": url, "content_type": "text/html", "text": "Longer evidence", "truncated": False,
    })
    client, calls = _client([
        _response({"content": None, "tool_calls": [_tool_call("search-1", "search_web", '{"query":"fact"}')]}, response_id="one", finish_reason="tool_calls"),
        _response({"content": None, "tool_calls": [_tool_call("fetch-1", "fetch_url", '{"url":"https://example.com/source"}')]}, response_id="two", finish_reason="tool_calls"),
        _response({"content": '{"answer":"42","confidence":0.9,"evidence":["Source"]}'}, response_id="three"),
    ])

    result = deepseek.ask_deepseek(client, _config(), "deepseek-v4-pro", "Solve", [{"type": "input_text", "text": "Question?"}])

    assert result["answer"] == "42"
    assert result["response_id"] == "three"
    assert result["usage"]["input_tokens"] == 30
    assert result["usage"]["output_tokens"] == 15
    assert result["tool_counts"]["total_tool_items"] == 2
    assert result["citations"] == ["https://example.com/source"]
    assert result["evidence_quality"]["fetched"] == 1
    assert result["evidence_quality"]["search_only"] == 0
    assert result["sources"][0]["excerpt"] == "Longer evidence"
    assert [item["tool"] for item in result["tool_trace"]] == ["search_web", "fetch_url"]
    assert len(result["tool_trace"][1]["sha256"]) == 64
    assert calls[1]["messages"][-1]["tool_call_id"] == "search-1"
    assert calls[2]["messages"][-1]["tool_call_id"] == "fetch-1"
    assert calls[0]["response_format"] == {"type": "json_object"}


def test_deepseek_inspects_requested_image_and_keeps_audit_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client, calls = _client([
        _response({"tool_calls": [_tool_call("visual-1", "inspect_visual", '{"locator":"prepared:0","question":"What color is the square?"}')]}, response_id="plan", finish_reason="tool_calls"),
        _response({"content": "A red square with no text."}, response_id="vision"),
        _response({"content": '{"answer":"red","confidence":0.8,"evidence":["image"]}'}, response_id="answer"),
    ])
    result = deepseek.ask_deepseek(client, _config(), "deepseek-v4-pro", "Solve", [
        {"type": "input_text", "text": "What color?"},
        {"type": "input_image", "image_url": _image_url()},
    ])

    assert [call["model"] for call in calls] == ["deepseek-v4-pro", "deepseek-flash", "deepseek-v4-pro"]
    assert "prepared:0" in calls[0]["messages"][1]["content"]
    assert calls[1]["messages"][1]["content"][0]["text"].startswith("Visual question: What color is the square?")
    assert calls[1]["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "A red square" in calls[2]["messages"][-1]["content"]
    assert "base64," not in calls[0]["messages"][1]["content"]
    assert result["vision_calls"] == 1
    assert result["vision_outputs"][0]["locator"] == "prepared:0"
    assert result["vision_outputs"][0]["asset_id"].startswith("visual:")
    assert len(result["vision_outputs"][0]["sha256"]) == 64
    assert result["vision_outputs"][0]["description"] == "A red square with no text."
    assert result["usage"]["total_tokens"] == 45
    assert result["diagnostics"]["model_calls"] == 3


def test_deepseek_reports_unusable_vision_evidence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client, calls = _client([
        _response({"tool_calls": [_tool_call("visual-1", "inspect_visual", '{"locator":"prepared:0","question":"What is shown?"}')]}, finish_reason="tool_calls"),
        _response({"content": "Unknown"}),
        _response({"content": '{"answer":"unknown","confidence":0.1,"evidence":[]}'}),
    ])
    result = deepseek.ask_deepseek(client, _config(), "deepseek-v4-pro", "Solve", [
        {"type": "input_text", "text": "Question?"},
        {"type": "input_image", "image_url": _image_url()},
    ])
    assert result["evidence_quality"]["tool_failures"] == 1
    assert "no usable image description" in calls[2]["messages"][-1]["content"]
    assert len(calls) == 3


def test_deepseek_rejects_too_many_images_before_spending_vision_calls():
    client, calls = _client([])
    content = [{"type": "input_text", "text": "Question?"}]
    content.extend({"type": "input_image", "image_url": "data:image/jpeg;base64,AAAA"} for _ in range(25))
    with pytest.raises(ValueError, match="Too many images"):
        deepseek.ask_deepseek(client, _config(), "deepseek-v4-pro", "Solve", content)
    assert calls == []


def test_deepseek_stops_vision_before_exceeding_task_token_budget(monkeypatch):
    monkeypatch.setattr(deepseek, "MAX_TASK_OUTPUT_TOKENS", 5)
    client, calls = _client([_response({"tool_calls": [_tool_call("visual-1", "inspect_visual", '{"locator":"prepared:0","question":"Count squares"}')]}, finish_reason="tool_calls", completion_tokens=5)])
    with pytest.raises(deepseek.DeepSeekAnswerError) as failure:
        deepseek.ask_deepseek(client, _config(), "deepseek-v4-pro", "Solve", [
            {"type": "input_text", "text": "How many squares?"},
            {"type": "input_image", "image_url": _image_url()},
        ])
    assert "visual_inspection_budget_exhausted" in str(failure.value)
    assert len(calls) == 1


def test_deepseek_finishes_after_tool_limit_without_executing_excess_calls(monkeypatch):
    queries = []
    monkeypatch.setattr(deepseek, "search_web", lambda query, **kwargs: queries.append(query) or {
        "provider": "bing_rss", "query": query, "results": [],
    })
    client, calls = _client([
        _response({"tool_calls": [
            _tool_call("one", "search_web", '{"query":"first"}'),
            _tool_call("two", "search_web", '{"query":"second"}'),
        ]}, finish_reason="tool_calls"),
        _response({"content": '{"answer":"42","confidence":0.4,"evidence":[]}'}, response_id="final"),
    ])
    result = deepseek.ask_deepseek(client, _config(max_tool_calls=1), "deepseek-v4-pro", "Solve", [
        {"type": "input_text", "text": "Question?"},
    ])
    assert queries == ["first"]
    assert result["response_id"] == "final"
    assert result["tool_counts"]["blocked_tool_calls"] == 1
    assert result["tool_counts"]["requested_tool_calls"] == 2
    assert result["diagnostics"]["budget_events"] == ["tool_call_limit"]
    assert result["diagnostics"]["blocked_reason"] == "tool_call_limit"
    assert result["tool_trace"][-1]["error"] == "tool_budget_exhausted"
    assert "tools" not in calls[1]


def test_deepseek_returns_search_failure_to_model(monkeypatch):
    def failed_search(*args, **kwargs):
        raise RuntimeError("Web search returned no usable results")

    monkeypatch.setattr(deepseek, "search_web", failed_search)
    client, calls = _client([
        _response({"tool_calls": [_tool_call("search", "search_web", '{"query":"fact"}')]}, finish_reason="tool_calls"),
        _response({"content": '{"answer":"42","confidence":0.5,"evidence":[]}'}),
    ])
    result = deepseek.ask_deepseek(client, _config(), "deepseek-v4-pro", "Solve", [
        {"type": "input_text", "text": "Question?"},
    ])
    assert result["answer"] == "42"
    assert "Web search returned no usable results" in calls[1]["messages"][-1]["content"]
    assert result["tool_trace"][0]["error"]
    assert result["evidence_quality"]["tool_failures"] == 1


def test_deepseek_can_recover_from_page_access_error(monkeypatch):
    monkeypatch.setattr(deepseek, "fetch_url", lambda url: {"url": url, "error": "HTTP 403"})
    client, calls = _client([
        _response({"tool_calls": [_tool_call("blocked", "fetch_url", '{"url":"https://example.com/blocked"}')]}, finish_reason="tool_calls"),
        _response({"content": '{"answer":"42","confidence":0.7,"evidence":["other source"]}'}),
    ])
    result = deepseek.ask_deepseek(client, _config(), "deepseek-v4-pro", "Solve", [
        {"type": "input_text", "text": "Question? https://example.com/blocked"},
    ])
    assert "HTTP 403" in calls[1]["messages"][-1]["content"]
    assert result["tool_trace"][0]["error"] == "HTTP 403"
    assert result["tool_counts"]["fetch_url_call"] == 1
    assert result["sources"][0]["status"] == "failed"
    assert result["evidence_quality"]["failed"] == 1


def test_deepseek_denies_unsearched_url_with_added_query(monkeypatch):
    monkeypatch.setattr(deepseek, "fetch_url", lambda url: pytest.fail("unapproved URL must not be fetched"))
    client, calls = _client([
        _response({"tool_calls": [_tool_call("blocked", "fetch_url", '{"url":"https://example.com/source?leak=attachment"}')]}, finish_reason="tool_calls"),
        _response({"content": '{"answer":"unknown","confidence":0.2,"evidence":[]}'}),
    ])
    result = deepseek.ask_deepseek(client, _config(), "deepseek-v4-pro", "Solve", [
        {"type": "input_text", "text": "Read https://example.com/source"},
    ])
    assert result["tool_trace"][0]["error"] == "unapproved_url"
    assert "leak=attachment" not in str(result["tool_trace"])
    assert result["tool_counts"]["blocked_tool_calls"] == 1
    assert result["diagnostics"]["blocked_reason"] == "unapproved_url"
    assert "unapproved_url" in calls[1]["messages"][-1]["content"]
    assert "tools" not in calls[1]


def test_finalization_prioritizes_fetched_sources_and_reports_omissions(monkeypatch):
    monkeypatch.setattr(deepseek, "MAX_FINAL_EVIDENCE_CHARS", 450)
    sources = {
        f"https://example.com/{index}": {"url": f"https://example.com/{index}", "title": "Search result", "status": "searched", "snippet": "old" * 100}
        for index in range(12)
    }
    sources["https://evidence.example/first"] = {"url": "https://evidence.example/first", "title": "First", "status": "fetched", "excerpt": "first fact"}
    sources["https://evidence.example/last"] = {"url": "https://evidence.example/last", "title": "Last", "status": "fetched", "excerpt": "decisive fact"}
    trace = [{"tool": "fetch_url", "url": "https://evidence.example/first", "chars": 100},
             {"tool": "fetch_url", "url": "https://evidence.example/last", "chars": 100}]
    prompt, quality = deepseek._final_input("Question?", sources, trace)
    assert "decisive fact" in prompt
    assert prompt.index("decisive fact") < prompt.index("first fact")
    assert "oldoldold" not in prompt
    assert quality["omitted_sources"] == []
    assert quality["evidence_degraded"] is False


def test_deepseek_deduplicates_tools_and_records_bounded_fetched_excerpt(monkeypatch):
    searches = []
    fetches = []
    monkeypatch.setattr(deepseek, "search_web", lambda query, **kwargs: searches.append(query) or {
        "provider": "bing_rss", "results": [{"title": "Source", "url": "https://example.com/fact", "snippet": "42 appears here"}],
    })
    monkeypatch.setattr(deepseek, "fetch_url", lambda url: fetches.append(url) or {
        "url": url, "content_type": "text/html", "text": "Introduction " + "filler " * 300 + "fact 42 confirmed", "truncated": True,
    })
    client, calls = _client([
        _response({"tool_calls": [_tool_call("s1", "search_web", '{"query":"fact"}')]}, finish_reason="tool_calls"),
        _response({"tool_calls": [_tool_call("s2", "search_web", '{"query":" FACT "}')]}, finish_reason="tool_calls"),
        _response({"tool_calls": [_tool_call("f1", "fetch_url", '{"url":"https://example.com/fact"}')]}, finish_reason="tool_calls"),
        _response({"tool_calls": [_tool_call("f2", "fetch_url", '{"url":"https://example.com/fact#section"}')]}, finish_reason="tool_calls"),
        _response({"content": '{"answer":"42","confidence":0.9,"evidence":["source"]}'}),
    ])
    result = deepseek.ask_deepseek(client, _config(max_tool_calls=5), "deepseek-v4-pro", "Solve", [{"type": "input_text", "text": "Question?"}])
    assert searches == ["fact"]
    assert fetches == ["https://example.com/fact"]
    assert result["tool_counts"]["cache_hits"] == 2
    assert result["evidence_quality"]["fetched"] == 1
    source = result["sources"][0]
    assert source["status"] == "fetched"
    assert source["truncated"] is True
    assert len(source["sha256"]) == 64
    assert len(source["excerpt"]) <= deepseek.MAX_SOURCE_EXCERPT_CHARS
    assert "fact 42 confirmed" in source["excerpt"]
    assert "Reuse the earlier search result" in calls[2]["messages"][-1]["content"]


def test_deepseek_finalizes_on_token_budget_with_no_tool_execution(monkeypatch):
    monkeypatch.setattr(deepseek, "MAX_TASK_INPUT_TOKENS", 5)
    monkeypatch.setattr(deepseek, "search_web", lambda *args, **kwargs: pytest.fail("tool should not execute"))
    client, calls = _client([
        _response({"tool_calls": [_tool_call("s1", "search_web", '{"query":"fact"}')]}, finish_reason="tool_calls"),
        _response({"content": '{"answer":"42","confidence":0.3,"evidence":[]}'}),
    ])
    result = deepseek.ask_deepseek(client, _config(), "deepseek-v4-pro", "Solve", [{"type": "input_text", "text": "Question?"}])
    assert result["answer"] == "42"
    assert result["tool_counts"]["blocked_tool_calls"] == 1
    assert "input_token_limit" in result["diagnostics"]["budget_events"]
    assert result["diagnostics"]["blocked_reason"] == "input_token_limit"
    assert "tools" not in calls[1]


def test_deepseek_recovers_from_length_and_empty_final_output():
    client, calls = _client([
        _response({"content": ""}, finish_reason="length"),
        _response({"content": ""}, finish_reason="stop"),
        _response({"content": '{"answer":"42","confidence":0.6,"evidence":[]}'}),
    ])
    result = deepseek.ask_deepseek(client, _config(), "deepseek-v4-pro", "Solve", [{"type": "input_text", "text": "Question?"}])
    assert result["answer"] == "42"
    assert result["diagnostics"]["final_attempts"] == 2
    assert "tools" not in calls[1] and "tools" not in calls[2]
    assert calls[2]["max_tokens"] > calls[1]["max_tokens"]


def test_deepseek_final_failure_exposes_only_safe_diagnostics():
    client, _ = _client([
        _response({"content": "private sk-secret123456789"}),
        _response({"content": ""}, finish_reason="length"),
        _response({"content": ""}),
    ])
    with pytest.raises(deepseek.DeepSeekAnswerError) as failure:
        deepseek.ask_deepseek(client, _config(), "deepseek-v4-pro", "Solve", [{"type": "input_text", "text": "Question?"}])
    assert failure.value.diagnostics["final_attempts"] == 2
    assert "sk-secret" not in str(failure.value)


def test_deepseek_time_budget_goes_directly_to_final_answer(monkeypatch):
    monkeypatch.setattr(deepseek, "MAX_TASK_SECONDS", 0)
    client, calls = _client([_response({"content": '{"answer":"42","confidence":0.4,"evidence":[]}'})])
    result = deepseek.ask_deepseek(client, _config(), "deepseek-v4-pro", "Solve", [{"type": "input_text", "text": "Question?"}])
    assert result["diagnostics"]["budget_events"] == ["wall_time_limit"]
    assert "tools" not in calls[0]


def test_deepseek_model_error_is_redacted_and_preserves_quota_status():
    class GatewayError(Exception):
        status_code = 402
        body = {"error": {"code": "insufficient_balance", "message": "sk-private123456789"}}

    client, _ = _client([GatewayError("request failed with sk-private123456789")])
    with pytest.raises(deepseek.DeepSeekCallError) as failure:
        deepseek.ask_deepseek(client, _config(), "deepseek-v4-pro", "Solve", [{"type": "input_text", "text": "Question?"}])
    assert failure.value.status_code == 402
    assert failure.value.code == "insufficient_balance"
    assert "sk-private" not in str(failure.value)


def test_deepseek_reuses_visual_observation_across_solver_and_judge(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    visual_call = _response({"tool_calls": [_tool_call("visual-1", "inspect_visual", '{"locator":"prepared:0","question":"What color is the square?"}')]}, finish_reason="tool_calls")
    content = [{"type": "input_text", "text": "What color?"}, {"type": "input_image", "image_url": _image_url()}]
    first_client, first_calls = _client([
        visual_call, _response({"content": "A red square."}),
        _response({"content": '{"answer":"red","confidence":0.7,"evidence":[]}'}),
    ])
    first = deepseek.ask_deepseek(first_client, _config(), "deepseek-v4-pro", "Solve", content)
    second_client, second_calls = _client([
        visual_call, _response({"content": '{"answer":"red","confidence":0.7,"evidence":[]}'}),
    ])
    second = deepseek.ask_deepseek(second_client, _config(), "deepseek-v4-pro", "Judge", content)
    assert first["vision_outputs"][0]["cache_hit"] is False
    assert second["vision_outputs"][0]["cache_hit"] is True
    assert second["vision_outputs"][0]["sha256"] == first["vision_outputs"][0]["sha256"]
    assert second["diagnostics"]["model_calls"] == 2
    assert len(first_calls) == 3 and len(second_calls) == 2


def test_deepseek_queries_only_current_attachment_and_calculates(tmp_path):
    table = tmp_path / "facts.csv"
    table.write_text("name,count\nA,2\nB,3\n", encoding="utf-8")
    client, calls = _client([
        _response({"tool_calls": [_tool_call("table", "query_attachment", '{"sql":"SELECT c1, c2 FROM sheet_1 WHERE row_number = 2"}')]}, finish_reason="tool_calls"),
        _response({"tool_calls": [_tool_call("math", "calculate", '{"expression":"2+3"}')]}, finish_reason="tool_calls"),
        _response({"content": '{"answer":"5","confidence":0.9,"evidence":["table calculation"]}'}),
    ])
    result = deepseek.ask_deepseek(
        client, _config(), "deepseek-v4-pro", "Solve",
        [{"type": "input_text", "text": "What is the sum?"}], attachment_path=table,
    )
    assert result["answer"] == "5"
    assert result["tool_trace"][0]["source_path"] == str(table.resolve())
    assert len(result["tool_trace"][0]["sha256"]) == 64
    assert result["tool_trace"][0]["row_count"] == 1
    assert result["tool_trace"][1]["result"] == "5"
    assert result["tool_counts"]["total_tool_items"] == 2
    assert '"c1": "A"' in calls[1]["messages"][-1]["content"]


def test_deepseek_rejects_model_supplied_attachment_path(tmp_path):
    table = tmp_path / "current.csv"
    table.write_text("current\n", encoding="utf-8")
    client, calls = _client([
        _response({"tool_calls": [_tool_call("table", "query_attachment", '{"path":"other.csv","sql":"SELECT * FROM sheet_1"}')]}, finish_reason="tool_calls"),
        _response({"content": '{"answer":"unknown","confidence":0.1,"evidence":[]}'}),
    ])
    result = deepseek.ask_deepseek(client, _config(), "deepseek-v4-pro", "Solve", [{"type": "input_text", "text": "Question?"}], attachment_path=table)
    assert result["tool_trace"][0]["error"] == "malformed_arguments"
    assert result["tool_counts"]["blocked_tool_calls"] == 1
    assert result["diagnostics"]["blocked_reason"] == "malformed_arguments"
    assert "malformed_arguments" in calls[1]["messages"][-1]["content"]
    assert "tools" not in calls[1]


def test_deepseek_shared_model_budget_counts_visual_call(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client, calls = _client([
        _response({"tool_calls": [_tool_call("visual", "inspect_visual", '{"locator":"prepared:0","question":"What color?"}')]}, finish_reason="tool_calls"),
    ])
    with pytest.raises(deepseek.DeepSeekAnswerError, match="visual_inspection_budget_exhausted"):
        deepseek.ask_deepseek(client, _config(), "deepseek-v4-pro", "Solve", [
            {"type": "input_text", "text": "What color?"},
            {"type": "input_image", "image_url": _image_url()},
        ], remaining_model_calls=2)
    assert len(calls) == 1


def test_deepseek_mid_run_failure_carries_cumulative_budget():
    client, calls = _client([
        _response({"tool_calls": [_tool_call("math", "calculate", '{"expression":"1+1"}')]}, finish_reason="tool_calls"),
        RuntimeError("gateway unavailable"),
    ])
    with pytest.raises(deepseek.DeepSeekCallError) as failure:
        deepseek.ask_deepseek(client, _config(), "deepseek-v4-pro", "Solve", [{"type": "input_text", "text": "Question?"}])
    assert failure.value.diagnostics["model_calls"] == 2
    assert failure.value.usage["input_tokens"] == 10
    assert failure.value.usage["output_tokens"] == 5
    assert len(calls) == 2


def test_deepseek_disables_repeated_degraded_search(monkeypatch):
    searches = []

    def degraded(query, **kwargs):
        searches.append(query)
        return {"provider": "none", "query": query, "results": [], "degraded_search": True,
                "warning": "Search coverage degraded", "backend_failures": {"bing_rss": "no_results"}}

    monkeypatch.setattr(deepseek, "search_web", degraded)
    client, calls = _client([
        _response({"tool_calls": [_tool_call("first", "search_web", '{"query":"same query"}')]}, finish_reason="tool_calls"),
        _response({"tool_calls": [_tool_call("again", "search_web", '{"query":"same query"}')]}, finish_reason="tool_calls"),
        _response({"content": '{"answer":"unknown","confidence":0.1,"evidence":[]}'}),
    ])
    result = deepseek.ask_deepseek(client, _config(max_tool_calls=5), "deepseek-v4-pro", "Solve", [{"type": "input_text", "text": "Question?"}])
    assert searches == ["same query"]
    assert result["evidence_quality"]["degraded_searches"] == 1
    assert result["evidence_quality"]["tool_failures"] == 1
    assert "Search backend remains degraded" in calls[2]["messages"][-1]["content"]
    assert all(tool["function"]["name"] != "search_web" for tool in calls[2]["tools"])


def test_deepseek_finalizes_on_disabled_search_before_other_tools(monkeypatch):
    searches = []

    def degraded(query, **kwargs):
        searches.append(query)
        return {"provider": "none", "query": query, "results": [], "degraded_search": True,
                "warning": "Search coverage degraded", "backend_failures": {"bing_rss": "no_results"}}

    monkeypatch.setattr(deepseek, "search_web", degraded)
    monkeypatch.setattr(deepseek, "calculate", lambda expression: pytest.fail("later tool must not execute"))
    client, calls = _client([
        _response({"tool_calls": [_tool_call("first", "search_web", '{"query":"first fact"}')]}, finish_reason="tool_calls"),
        _response({"tool_calls": [_tool_call("second", "search_web", '{"query":"second fact"}')]}, finish_reason="tool_calls"),
        _response({"tool_calls": [
            _tool_call("blocked", "search_web", '{"query":"third fact"}'),
            _tool_call("later", "calculate", '{"expression":"1+1"}'),
        ]}, finish_reason="tool_calls"),
        _response({"content": '{"answer":"unknown","confidence":0.1,"evidence":[]}'}, response_id="final"),
    ])
    result = deepseek.ask_deepseek(client, _config(max_tool_calls=8), "deepseek-v4-pro", "Solve", [
        {"type": "input_text", "text": "Question?"},
    ])
    assert searches == ["first fact", "second fact"]
    assert len(calls) == 4
    assert all(tool["function"]["name"] != "search_web" for tool in calls[2]["tools"])
    assert "tools" not in calls[3]
    assert result["response_id"] == "final"
    assert result["tool_counts"]["requested_tool_calls"] == 3
    assert result["tool_counts"]["blocked_tool_calls"] == 1
    assert result["tool_counts"]["calculate_call"] == 0
    assert result["tool_trace"][-1]["error"] == "degraded_search_disabled"
    assert result["diagnostics"]["blocked_reason"] == "degraded_search_disabled"
    assert result["diagnostics"]["final_attempts"] == 1
    assert result["diagnostics"]["model_calls"] == 4
    assert result["evidence_quality"]["degraded_searches"] == 2
    assert result["evidence_quality"]["tool_failures"] == 2
    assert "Search coverage degraded" in calls[3]["messages"][-1]["content"]
    assert "degraded_search_disabled" in calls[3]["messages"][-1]["content"]


def test_deepseek_fetches_trusted_direct_url_after_degraded_search(monkeypatch):
    url = "https://en.wikipedia.org/wiki/Example"
    monkeypatch.setattr(deepseek, "search_web", lambda query, **kwargs: {
        "provider": "none", "query": query, "results": [], "degraded_search": True,
        "warning": "Search coverage degraded",
    })
    fetched = []
    monkeypatch.setattr(deepseek, "fetch_url", lambda candidate: fetched.append(candidate) or {
        "url": candidate, "content_type": "text/html", "text": "Evidence from the direct page", "truncated": False,
    })
    client, calls = _client([
        _response({"tool_calls": [_tool_call("search", "search_web", '{"query":"Example"}')]}, finish_reason="tool_calls"),
        _response({"tool_calls": [_tool_call("direct", "fetch_url", '{"url":"' + url + '"}')]}, finish_reason="tool_calls"),
        _response({"content": '{"answer":"Example","confidence":0.7,"evidence":["Wikipedia"]}'}),
    ])
    result = deepseek.ask_deepseek(client, _config(max_tool_calls=5), "deepseek-v4-pro", "Solve", [
        {"type": "input_text", "text": "Question?"},
    ])
    assert fetched == [url]
    assert result["sources"][0]["provenance"] == "direct_fetch"
    assert result["sources"][0]["status"] == "fetched"
    assert result["tool_trace"][1]["provenance"] == "direct_fetch"
    assert result["evidence_quality"]["degraded_searches"] == 1
    assert len(calls) == 3


def test_deepseek_blocks_third_direct_url_and_skips_later_tool(monkeypatch):
    urls = [f"https://en.wikipedia.org/wiki/Example_{index}" for index in range(3)]
    monkeypatch.setattr(deepseek, "search_web", lambda query, **kwargs: {
        "provider": "none", "query": query, "results": [], "degraded_search": True,
        "warning": "Search coverage degraded",
    })
    fetched = []
    monkeypatch.setattr(deepseek, "fetch_url", lambda candidate: fetched.append(candidate) or {
        "url": candidate, "content_type": "text/html", "text": "Page evidence", "truncated": False,
    })
    monkeypatch.setattr(deepseek, "calculate", lambda expression: pytest.fail("later tool must not execute"))
    client, calls = _client([
        _response({"tool_calls": [_tool_call("search", "search_web", '{"query":"Examples"}')]}, finish_reason="tool_calls"),
        _response({"tool_calls": [_tool_call("first", "fetch_url", '{"url":"' + urls[0] + '"}')]}, finish_reason="tool_calls"),
        _response({"tool_calls": [_tool_call("second", "fetch_url", '{"url":"' + urls[1] + '"}')]}, finish_reason="tool_calls"),
        _response({"tool_calls": [
            _tool_call("third", "fetch_url", '{"url":"' + urls[2] + '"}'),
            _tool_call("later", "calculate", '{"expression":"1+1"}'),
        ]}, finish_reason="tool_calls"),
        _response({"content": '{"answer":"unknown","confidence":0.1,"evidence":[]}'}, response_id="final"),
    ])
    result = deepseek.ask_deepseek(client, _config(max_tool_calls=8), "deepseek-v4-pro", "Solve", [
        {"type": "input_text", "text": "Question?"},
    ])
    assert fetched == urls[:2]
    assert len(calls) == 5 and "tools" not in calls[4]
    assert result["tool_counts"]["requested_tool_calls"] == 4
    assert result["tool_counts"]["blocked_tool_calls"] == 1
    assert result["tool_counts"]["calculate_call"] == 0
    assert result["tool_trace"][-1]["error"] == "direct_fetch_limit"
    assert result["diagnostics"]["blocked_reason"] == "direct_fetch_limit"


@pytest.mark.parametrize("url", [
    "https://en.wikipedia.org/wiki/Example?token=secret",
    "https://example.com/article",
    "http://127.0.0.1/private",
    "https://wikipedia.org.evil.com/article",
    "https://en.wikipedia.org/" + "a" * 300,
    "http://[not-valid/",
])
def test_deepseek_blocks_unsafe_direct_urls_after_degraded_search(monkeypatch, url):
    monkeypatch.setattr(deepseek, "search_web", lambda query, **kwargs: {
        "provider": "none", "query": query, "results": [], "degraded_search": True,
        "warning": "Search coverage degraded",
    })
    monkeypatch.setattr(deepseek, "fetch_url", lambda candidate: pytest.fail("unsafe URL must not be fetched"))
    client, calls = _client([
        _response({"tool_calls": [_tool_call("search", "search_web", '{"query":"Example"}')]}, finish_reason="tool_calls"),
        _response({"tool_calls": [_tool_call("unsafe", "fetch_url", '{"url":"' + url + '"}')]}, finish_reason="tool_calls"),
        _response({"content": '{"answer":"unknown","confidence":0.1,"evidence":[]}'}),
    ])
    result = deepseek.ask_deepseek(client, _config(max_tool_calls=5), "deepseek-v4-pro", "Solve", [
        {"type": "input_text", "text": "Question?"},
    ])
    assert result["tool_trace"][-1]["error"] == "unapproved_url"
    assert result["diagnostics"]["blocked_reason"] == "unapproved_url"
    assert result["tool_counts"]["blocked_tool_calls"] == 1
    assert "tools" not in calls[2]


def test_deepseek_stops_when_direct_fetch_rejects_nonpublic_resolution(monkeypatch):
    monkeypatch.setattr(deepseek, "search_web", lambda query, **kwargs: {
        "provider": "none", "query": query, "results": [], "degraded_search": True,
        "warning": "Search coverage degraded",
    })
    monkeypatch.setattr(deepseek, "fetch_url", lambda url: (_ for _ in ()).throw(ValueError("Nonpublic URLs are not allowed")))
    client, calls = _client([
        _response({"tool_calls": [_tool_call("search", "search_web", '{"query":"Example"}')]}, finish_reason="tool_calls"),
        _response({"tool_calls": [_tool_call("unsafe", "fetch_url", '{"url":"https://en.wikipedia.org/wiki/Example"}')]}, finish_reason="tool_calls"),
        _response({"content": '{"answer":"unknown","confidence":0.1,"evidence":[]}'}),
    ])
    result = deepseek.ask_deepseek(client, _config(max_tool_calls=5), "deepseek-v4-pro", "Solve", [
        {"type": "input_text", "text": "Question?"},
    ])
    assert result["tool_trace"][-1]["error"] == "unsafe_direct_url"
    assert result["tool_counts"]["blocked_tool_calls"] == 1
    assert result["diagnostics"]["blocked_reason"] == "unsafe_direct_url"
    assert "tools" not in calls[2]


def test_bing_rss_search_parses_sources_without_credentials(monkeypatch):
    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b'<rss><channel><item><title>Example</title><link>https://example.com/</link><description>Useful fact</description></item></channel></rss>'

    monkeypatch.setattr(web_tools.requests, "get", lambda *args, **kwargs: Response())
    result = web_tools.search_web("sample query")
    assert result["provider"] == "bing_rss"
    assert result["results"] == [{"title": "Example", "url": "https://example.com/", "snippet": "Useful fact"}]


def test_fetch_rejects_private_urls():
    with pytest.raises(ValueError, match="Nonpublic"):
        web_tools.fetch_url("http://127.0.0.1/private")


def test_fetch_returns_http_access_error_for_model_recovery(monkeypatch):
    class Forbidden:
        status_code = 403
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(web_tools.socket, "getaddrinfo", lambda *args: [(None, None, None, None, ("93.184.215.14", 443))])
    monkeypatch.setattr(web_tools.requests, "get", lambda *args, **kwargs: Forbidden())
    assert web_tools.fetch_url("https://example.com/") == {"url": "https://example.com/", "error": "HTTP 403"}


def test_search_timeout_is_explicit(monkeypatch):
    def timeout(*args, **kwargs):
        raise requests.Timeout("unavailable")

    monkeypatch.setattr(web_tools.requests, "get", timeout)
    result = web_tools.search_web("sample query")
    assert result["degraded_search"] is True
    assert result["backend_failures"]["bing_rss"] == "Timeout"
    assert "bing_rss:Timeout" in result["warning"]


def test_fetch_rechecks_redirect_target(monkeypatch):
    class Redirect:
        status_code = 302
        headers = {"Location": "http://127.0.0.1/private"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    requested = []
    monkeypatch.setattr(web_tools.socket, "getaddrinfo", lambda *args: [(None, None, None, None, ("93.184.215.14", 443))])
    monkeypatch.setattr(web_tools.requests, "get", lambda url, **kwargs: requested.append(url) or Redirect())
    with pytest.raises(ValueError, match="Nonpublic"):
        web_tools.fetch_url("https://example.com/start")
    assert requested == ["https://example.com/start"]
