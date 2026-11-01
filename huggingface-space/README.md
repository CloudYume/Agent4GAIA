---
title: Agent4GAIA
sdk: gradio
app_file: app.py
---

# Agent4GAIA Space

A manual question-and-attachment interface for [Agent4GAIA](https://github.com/CloudYume/Agent4GAIA). Enter the configured access token, a question, its level, and an optional attachment, then choose **Solve**. The app returns the final answer, confidence, and evidence or warnings.

## Required configuration

Set `SPACE_ACCESS_TOKEN` and the selected model-provider keys as Hugging Face Space Secrets. The default configuration uses `DEEPSEEK_API_KEY` and `ANTHROPIC_API_KEY`. Model IDs and endpoint support depend on your provider. The access token entered in the app is not a model-provider key or Hugging Face token.

## Build and deployment

This source directory is part of a larger repository. Build the complete deployment from the repository root with `python scripts/build_space.py --output dist/space-bundle`, then upload the bundle contents to the Space root. See the [deployment guide](https://github.com/CloudYume/Agent4GAIA/blob/main/docs/space-deployment.md) for setup.

This Space does not contain GAIA benchmark questions, attachments, or answers. It accepts user-supplied material only; solving may make paid model calls. No official GAIA score is claimed.
