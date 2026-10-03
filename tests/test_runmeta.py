from dataclasses import replace
from pathlib import Path

import pytest

from gaia_agent import runmeta
from gaia_agent.config import load_config
from gaia_agent.sources import Task


def test_implementation_change_changes_run_signature(monkeypatch):
    config = load_config("config.example.toml")
    original = runmeta.run_signature(config)
    monkeypatch.setattr(runmeta, "_implementation_hash", lambda: "changed")
    assert runmeta.run_signature(config) != original


@pytest.mark.parametrize("name", ["cli.py", "validation.py", "store.py", "scoring.py"])
def test_implementation_hash_covers_run_and_export_code(monkeypatch, name):
    original = runmeta._implementation_hash()
    read_bytes = Path.read_bytes

    def changed(path):
        content = read_bytes(path)
        return content + b"changed" if path.name == name else content

    monkeypatch.setattr(Path, "read_bytes", changed)
    assert runmeta._implementation_hash() != original


def test_input_fingerprint_changes_with_question_and_attachment_bytes(tmp_path):
    attachment = tmp_path / "evidence.txt"
    attachment.write_text("first", encoding="utf-8")
    task = Task("abc", "Original question?", 2, file_name="evidence.txt", attachment_path=str(attachment))
    original = runmeta.input_fingerprint(task)
    task.question = "Revised question?"
    assert runmeta.input_fingerprint(task) != original
    task.question = "Original question?"
    attachment.write_text("second", encoding="utf-8")
    assert runmeta.input_fingerprint(task) != original


def test_input_fingerprint_requires_attachment_bytes():
    with pytest.raises(RuntimeError, match="Missing attachment"):
        runmeta.input_fingerprint(Task("abc", "Question?", 1, file_name="evidence.txt"))

def test_code_interpreter_setting_changes_run_signature():
    config = load_config("config.example.toml")
    assert runmeta.run_signature(config) != runmeta.run_signature(replace(config, code_interpreter_enabled=True))


@pytest.mark.parametrize("setting,value", [
    ("search_context_size", "high"),
    ("max_tool_calls", 0),
    ("max_tool_calls", 4),
    ("max_output_tokens", 2000),
])
def test_request_budget_setting_changes_run_signature(setting, value):
    config = load_config("config.example.toml")
    assert runmeta.run_signature(config) != runmeta.run_signature(replace(config, **{setting: value}))


def test_code_interpreter_setting_requires_toml_boolean(tmp_path):
    source = Path("config.example.toml").read_text(encoding="utf-8")
    target = tmp_path / "bad.toml"
    target.write_text(source.replace("code_interpreter_enabled = false", 'code_interpreter_enabled = "false"'), encoding="utf-8")
    with pytest.raises(ValueError, match="TOML boolean"):
        load_config(target)


@pytest.mark.parametrize("setting,invalid,error", [
    ("search_context_size", "\"wide\"", "search_context_size"),
    ("search_context_size", "[\"low\"]", "search_context_size"),
    ("max_tool_calls", "-1", "max_tool_calls"),
    ("max_tool_calls", "true", "max_tool_calls"),
    ("max_output_tokens", "-1", "max_output_tokens"),
    ("max_output_tokens", "\"8000\"", "max_output_tokens"),
])
def test_request_budget_rejects_invalid_values(tmp_path, setting, invalid, error):
    source = Path("config.example.toml").read_text(encoding="utf-8")
    default = {"search_context_size": '"medium"', "max_tool_calls": "12", "max_output_tokens": "8000"}[setting]
    target = tmp_path / "bad.toml"
    target.write_text(source.replace(f"{setting} = {default}", f"{setting} = {invalid}"), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_config(target)


def test_zero_max_tool_calls_is_valid(tmp_path):
    source = Path("config.example.toml").read_text(encoding="utf-8")
    target = tmp_path / "compatible.toml"
    target.write_text(source.replace("max_tool_calls = 12", "max_tool_calls = 0"), encoding="utf-8")
    assert load_config(target).max_tool_calls == 0
