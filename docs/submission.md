# GAIA Submission Guide

[Project overview](../README.md) · [Configuration template](../config.example.toml)

This project supports two separate destinations: the Agents Course Unit 4 scoring service and the official GAIA leaderboard. A course score is not an official GAIA score. The supported official export is the complete GAIA 2023 test set; this repository does not provide a verified answer file or claim a leaderboard result.

## Course submission

Complete the course's 20 tasks, then export a course payload:

```powershell
python -m gaia_agent.cli export course --runs .runs/course --output .runs/course-payload.json --username YOUR_HF_USERNAME --agent-code https://huggingface.co/spaces/YOUR_HF_USERNAME/YOUR_SPACE/tree/main
```

The code URL must point to an existing public Hugging Face Space with inspectable agent code. See [Space deployment](space-deployment.md) for an allowlisted bundle. Inspect the exported payload before calling `python -m gaia_agent.cli submit-course .runs/course-payload.json`; that command sends a real submission to the course service.

## Official evaluation workflow

Run from the checkout root in your configured Python environment. GAIA requires a Hugging Face account that has accepted the dataset's gated terms. Validation has reference answers; test answers are withheld.

### 1. Complete the validation pilot

```powershell
python -m gaia_agent.cli preflight validation --runs .runs/validation-pilot
python -m gaia_agent.cli run validation --runs .runs/validation-pilot --manifest .runs/validation-pilot/.meta/pilot-20.json
python -m gaia_agent.cli score-validation validation --runs .runs/validation-pilot --manifest .runs/validation-pilot/.meta/pilot-20.json --output .runs/validation-pilot/.meta/score.json
```

The fixed pilot selects 20 questions with Level 1/2/3 quotas of 6/10/4 and covers available evidence types. Check `pilot_complete` and `meets_40_percent_gate` in the score report. Completing all 20 questions with at least 40% local accuracy is the project's recommended readiness check, not an official GAIA requirement or an automatically enforced prerequisite for `run test`.

`preflight` checks access, attachments, local audio decoding, search health, and input fingerprints without calling the solver. Degraded search is recorded separately and does not by itself block a pilot. Preflight may still access network services and download gated input files.

### 2. Run the complete test set

```powershell
python -m gaia_agent.cli preflight test --runs .runs/test-official
python -m gaia_agent.cli run test --runs .runs/test-official
python -m gaia_agent.cli status test --runs .runs/test-official
python -m gaia_agent.cli export test --runs .runs/test-official --output .runs/gaia-test.jsonl
python -m gaia_agent.cli check-official .runs/gaia-test.jsonl
```

Use a new run directory for test. A run is bound to its code, configuration, task manifest, and attachment fingerprints. Resume unchanged inputs in the same directory; changed inputs require another preflight and a new directory.

Strict export rejects incomplete tasks, blank or multiline answers, unresolved review disputes, and unreviewed evidence problems. Inspect the individual task record and original evidence before retrying. The advanced `--review-waivers` option accepts only supported, per-task exceptions bound to the exact task, answer, run, input, and evidence fingerprints; it cannot create missing answers or waive every error. Its implementation is in [`validation.py`](../gaia_agent/validation.py) and [`cli.py`](../gaia_agent/cli.py).

### 3. Inspect the answer file

The output is UTF-8 JSON Lines: exactly 301 unique test task IDs, with Level 1/2/3 counts of 93/159/49. Each line has `task_id` and `model_answer`. The optional `reasoning_trace` is intentionally omitted.

```jsonl
{"task_id":"<test-task-id-1>","model_answer":"<short-answer-1>"}
{"task_id":"<test-task-id-2>","model_answer":"<short-answer-2>"}
```

These lines illustrate the schema; they are not a submission. Keep answers to the requested number, words, or ordered list. Omit explanations and the `FINAL ANSWER:` prefix. Check dates, units, ordering, and source conflicts against each question. The private `.meta/result.json` progress report is not an answer file, and passing `check-official` validates structure and coverage rather than answer correctness.

## Leaderboard form

Upload only the validated answer file at the [GAIA leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard). The CLI does not submit official results. The form's live labels and eligibility rules can change; check the current [form implementation](https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/app.py) before uploading.

| Field | What to provide |
| --- | --- |
| Agent name | Your actual agent name and the evaluated version. |
| Model family | The solver, adjudicator, reviewer, and any visual or transcription models actually used. |
| System prompt example | The evaluated prompt, including provider-specific output constraints; do not substitute the leaderboard's sample prompt. |
| Model or project URL | An existing, publicly accessible project, Space, or model-information page, such as [Agent4GAIA](https://github.com/CloudYume/Agent4GAIA). |
| Organisation | Your public organization name, or `Independent` when appropriate. |
| Contact email | A working address; check the live form's handling of this field. |
| File | The validated `.runs/gaia-test.jsonl`, not a run directory, configuration file, or deployment bundle. |

The template currently selects DeepSeek for solving/adjudication, DeepSeek Flash for image preprocessing, and Anthropic for independent review. Use actual run metadata rather than assuming those defaults were used. Prompt sources are [`agent.py`](../gaia_agent/agent.py), [`deepseek.py`](../gaia_agent/deepseek.py), and [`reviewer.py`](../gaia_agent/reviewer.py); the optional OpenAI Responses route has different tool behavior.

Complete the Space's own Hugging Face sign-in, fill the fields, attach the validated JSONL, and review everything before choosing **Submit Eval On Test**. Being signed in to the Hugging Face website may not complete the embedded Space's OAuth flow. Submission quotas and account-age checks belong to the live service; failed uploads may consume an attempt.

Keep API keys, pairing/access tokens, private run records, gated questions, attachments, and answer files out of the public repository and Space bundle.
