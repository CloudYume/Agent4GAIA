"""公开 Space 的手动问答入口，不加载或发布 GAIA 私有题目。"""

from __future__ import annotations

import os
from hmac import compare_digest
from pathlib import Path
from uuid import uuid4

import gradio as gr

from gaia_agent.agent import GaiaAgent
from gaia_agent.config import load_config
from gaia_agent.sources import Task


def solve(access_token: str, question: str, level: int, attachment: str | None) -> tuple[str, str, str]:
    expected = os.getenv("SPACE_ACCESS_TOKEN", "")
    if not expected or not compare_digest(access_token or "", expected):
        raise gr.Error("Access denied")
    if not question.strip():
        raise gr.Error("Question is required")
    path = Path(attachment) if attachment else None
    task = Task(
        task_id=str(uuid4()),
        question=question.strip(),
        level=int(level),
        file_name=path.name if path else "",
        attachment_path=str(path) if path else "",
    )
    try:
        result = GaiaAgent(load_config()).solve(task)
    except Exception as exc:
        raise gr.Error(f"Agent failed: {type(exc).__name__}") from exc
    final = result["final"]
    evidence = final.get("citations") or final.get("evidence") or []
    warnings = result.get("warnings") or []
    return (
        result["answer"],
        f"{result['confidence']:.2f}",
        "\n".join(str(item) for item in [*evidence, *warnings]),
    )


with gr.Blocks(title="GAIA Course Agent") as demo:
    gr.Markdown("# GAIA Course Agent")
    access_token = gr.Textbox(label="Access token", type="password")
    question = gr.Textbox(label="Question", lines=5)
    level = gr.Dropdown([1, 2, 3], value=1, label="Level")
    attachment = gr.File(label="Attachment", type="filepath")
    submit = gr.Button("Solve", variant="primary")
    answer = gr.Textbox(label="Answer")
    confidence = gr.Textbox(label="Confidence")
    evidence = gr.Textbox(label="Evidence and warnings", lines=6)
    submit.click(solve, inputs=[access_token, question, level, attachment], outputs=[answer, confidence, evidence], concurrency_limit=1)


if __name__ == "__main__":
    demo.launch()
