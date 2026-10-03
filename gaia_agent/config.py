"""读取本地配置，并允许环境变量覆盖密钥与网关地址。"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    openai_key: str
    openai_base_url: str
    deepseek_key: str
    deepseek_base_url: str
    anthropic_key: str
    anthropic_base_url: str
    hf_token: str
    solver_provider: str
    solver_model: str
    reviewer_provider: str
    reviewer_model: str
    judge_provider: str
    judge_model: str
    transcription_model: str
    transcription_provider: str
    local_asr_model: str
    youtube_cookies_from_browser: str
    reasoning_effort: str
    code_interpreter_enabled: bool
    search_context_size: str
    max_tool_calls: int
    max_output_tokens: int
    review_mode: str
    confidence_threshold: float
    workers: int

    def require(self, provider: str) -> None:
        if provider == "openai" and not self.openai_key:
            raise RuntimeError("Set [openai].api_key in config.toml or OPENAI_API_KEY")
        if provider == "deepseek" and not self.deepseek_key:
            raise RuntimeError("Set [deepseek].api_key in config.toml or DEEPSEEK_API_KEY")
        if provider == "anthropic" and not self.anthropic_key:
            raise RuntimeError("Set [anthropic].api_key in config.toml or ANTHROPIC_API_KEY")
        if provider not in {"openai", "deepseek", "anthropic"}:
            raise ValueError(f"Unsupported provider: {provider}")


def load_config(path: str | Path = "config.toml") -> Config:
    config_path = Path(path)
    if not config_path.exists():
        if config_path.name == "config.toml" and Path("config.example.toml").exists():
            config_path = Path("config.example.toml")
        else:
            raise FileNotFoundError(f"Missing {config_path}; create it from config.example.toml")
    with config_path.open("rb") as handle:
        values = tomllib.load(handle)
    agent = values.get("agent", {})
    code_interpreter_enabled = agent.get("code_interpreter_enabled", True)
    if not isinstance(code_interpreter_enabled, bool):
        raise ValueError("code_interpreter_enabled must be a TOML boolean")
    search_context_size = agent.get("search_context_size", "medium")
    max_tool_calls = agent.get("max_tool_calls", 12)
    max_output_tokens = agent.get("max_output_tokens", 8000)
    youtube_cookies_from_browser = agent.get("youtube_cookies_from_browser", "")
    if not isinstance(search_context_size, str) or search_context_size not in {"low", "medium", "high"}:
        raise ValueError("search_context_size must be low, medium, or high")
    if type(max_tool_calls) is not int or max_tool_calls < 0:
        raise ValueError("max_tool_calls must be a nonnegative integer (0 omits the parameter)")
    if type(max_output_tokens) is not int or max_output_tokens < 1:
        raise ValueError("max_output_tokens must be a positive integer")
    if not isinstance(youtube_cookies_from_browser, str):
        raise ValueError("youtube_cookies_from_browser must be a string")
    result = Config(
        openai_key=os.getenv("OPENAI_API_KEY") or values.get("openai", {}).get("api_key", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL") or values.get("openai", {}).get("base_url", ""),
        deepseek_key=os.getenv("DEEPSEEK_API_KEY") or values.get("deepseek", {}).get("api_key", ""),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL") or values.get("deepseek", {}).get("base_url", "https://api.deepseek.com"),
        anthropic_key=os.getenv("ANTHROPIC_API_KEY") or values.get("anthropic", {}).get("api_key", ""),
        anthropic_base_url=os.getenv("ANTHROPIC_BASE_URL") or values.get("anthropic", {}).get("base_url", ""),
        hf_token=os.getenv("HF_TOKEN") or values.get("huggingface", {}).get("token", ""),
        solver_provider=agent.get("solver_provider", "deepseek"),
        solver_model=agent.get("solver_model", "deepseek-v4-pro"),
        reviewer_provider=agent.get("reviewer_provider", "anthropic"),
        reviewer_model=agent.get("reviewer_model", "claude-opus-5-5"),
        judge_provider=agent.get("judge_provider", "deepseek"),
        judge_model=agent.get("judge_model", "deepseek-v4-pro"),
        transcription_model=agent.get("transcription_model", "gpt-transcribe"),
        transcription_provider=agent.get("transcription_provider", "local"),
        local_asr_model=agent.get("local_asr_model", "small"),
        youtube_cookies_from_browser=youtube_cookies_from_browser,
        reasoning_effort=agent.get("reasoning_effort", "high"),
        code_interpreter_enabled=code_interpreter_enabled,
        search_context_size=search_context_size,
        max_tool_calls=max_tool_calls,
        max_output_tokens=max_output_tokens,
        review_mode=agent.get("review_mode", "adaptive"),
        confidence_threshold=float(agent.get("confidence_threshold", 0.8)),
        workers=int(agent.get("workers", 3)),
    )
    if result.review_mode not in {"off", "adaptive", "always"}:
        raise ValueError("review_mode must be off, adaptive, or always")
    if not 0 <= result.confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if result.workers < 1:
        raise ValueError("workers must be at least 1")
    if result.transcription_provider not in {"local", "openai"}:
        raise ValueError("transcription_provider must be local or openai")
    return result
