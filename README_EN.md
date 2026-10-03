# GAIA Course Agent

[中文文档](README.md)

A resumable CLI for GAIA questions. One solver and review pipeline supports two **distinct** evaluations. A course score cannot be submitted as an official GAIA leaderboard score.

| Track | Questions | Submission | Destination |
| --- | --- | --- | --- |
| Hugging Face Agents Course Unit 4 | 20 selected GAIA validation Level 1 questions | `username`, public Space code URL, 20 `submitted_answer` values | [Course scoring API](https://agents-course-unit4-scoring.hf.space/docs) and [student leaderboard](https://huggingface.co/spaces/agents-course/Students_leaderboard); the introduction says 30%, while the certificate page says above 30%, so aim for at least 35% |
| Official GAIA leaderboard | 301 questions in the 2023 test split: 93/159/49 at Levels 1/2/3 | 301 JSONL lines, each with `task_id` and `model_answer` | [GAIA leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard) upload form; test answers are private |

The [GAIA dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA) is gated. Sign in and accept its terms before accessing it. Do not publish validation or test questions, attachments, answers, or run records in a public Space, Git repository, or other crawlable location.

## Course units and implementation

| Course unit | Implementation in this project |
| --- | --- |
| Unit 1: agent loop and tools | `gaia_agent/agent.py` provides solver prompts and structured answers; the function-tool loop in `deepseek.py` and per-task records support thought, action, and observation. |
| Unit 2: frameworks, multi-agent, and visual browsing | `orchestrator.py` coordinates a DeepSeek V4 Pro solver, a blind Claude Opus 5.5 reviewer, and a Pro dispute judge. `memory.py` keeps a per-task evidence ledger and `contracts.py` enforces the shared budget. `attachments.py` and `media.py` expose media locators; the model calls DeepSeek Flash through `inspect_visual` only when a visual observation is needed. This follows the course's orchestration ideas without depending directly on smolagents. |
| Unit 3: Agentic RAG | `web_tools.py` aggregates public search backends and marks weak coverage; `sources.py` / `attachments.py` extract local evidence, while `tools/data.py` provides read-only table SQL and bounded arithmetic. There is no separate vector index. |
| Unit 4: GAIA project and evaluation | `cli.py`, `store.py`, `scoring.py`, and `validation.py` provide stratified pilots, resumable per-task runs, input snapshots, known-answer validation scoring, and official submission checks. |

## Install and configure

Use Python 3.11+. From the project root in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[audio,video,dev]"
Copy-Item config.example.toml config.toml
.\.venv\Scripts\gaia-agent doctor
```

The `audio` extra installs local CPU Whisper transcription, required by the default `transcription_provider = "local"`. With `"openai"` transcription, you can install only `".[video,dev]"` and configure a transcription model. The `video` extra installs `yt-dlp` and `imageio-ffmpeg` for YouTube URLs, frame sampling, and audio extraction; downloads still depend on network and site availability. `[agent] youtube_cookies_from_browser` defaults to an empty string, so browser cookies are never read implicitly. Set it to `"edge:Default"` only when you explicitly intend to use that signed-in Edge profile. A running browser may lock its cookie database, and closing it may still leave app-bound encryption unsupported. Keep exported cookies out of source control and logs. `dev` is only for local tests.

Fill your private `config.toml` with working model IDs and credentials. By default, the solver and dispute judge use `deepseek-v4-pro` through the official DeepSeek Chat Completions API. The same DeepSeek credential calls `deepseek-flash` only when the model needs to inspect an image or video frame; Pro itself does not accept image input. The independent reviewer uses Claude Opus 5.5 through the Anthropic Messages API and can use its server-side web search. A local function-tool loop aggregates Bing RSS, Bing HTML, Wikipedia, and optionally Tavily when `TAVILY_API_KEY` is set. Search and page retrieval require network access. When no independent relevant source is available, the result is marked degraded rather than treating a single Wikipedia hit as sufficient evidence. The current network may expose only weak public search results; preflight records that limitation and the pilot can continue, with affected answers requiring evidence review. DeepSeek's Responses API ignores built-in `web_search` and `code_interpreter`, so changing only the model name in an old OpenAI configuration is insufficient. `code_interpreter_enabled = false` is the DeepSeek default; this switch controls built-in Code Interpreter only when the optional OpenAI Responses solver is selected. `review_mode = "off"` requires only solver credentials; `"adaptive"` or `"always"` also requires reviewer credentials. `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, and `HF_TOKEN` override the file. The optional OpenAI solver instead uses `OPENAI_API_KEY` and `OPENAI_BASE_URL`. `HF_TOKEN` accesses the gated dataset and provides a fallback for course attachments; the course question list is public. `doctor` prints a configuration summary and whether credentials are present, without calling models.

Under `[agent]`, `search_context_size = "medium"` controls only the optional OpenAI Responses built-in web search context; local tools bound DeepSeek search result count and text length. For DeepSeek, `max_tool_calls = 12` limits function calls executed by the local host during each solve or adjudication (hard cap 12; `0` still uses the internal default budget of 8). With OpenAI Responses, a positive value is sent to the API and `0` omits that request field. DeepSeek's Responses API ignores `max_tool_calls`, so its budget must be enforced by the local loop. `max_output_tokens = 8000` limits each model request's output tokens, including reasoning tokens; independent Anthropic review has its own limit. These are per-question/per-request budgets, not a hard cap on total batch cost: search, Flash image preprocessing, retries, reviews, and adjudications add usage. A very low output limit may prevent a complete answer.

See [config.example.toml](config.example.toml) for each setting. `workers` sets question concurrency and affects cost and rate limits. `confidence_threshold` applies only to `adaptive` review. Configuration, prompts, and hashes of key implementation source files form the run signature. Official test runs also bind the full task and attachment inputs in `.meta/run-snapshot.json`. If code, config, or inputs change, start a fresh `--runs` directory; `--force` does not bypass the snapshot.

The commands below assume an activated virtual environment. Otherwise replace `gaia-agent` with `.\.venv\Scripts\gaia-agent`. Put the global option before the subcommand when selecting another config: `gaia-agent --config my-config.toml doctor`.

## Course: 20 questions

Start with `--limit 1` to check the entire solving path, then resume the remaining questions. Completed tasks are skipped by default. Each result is saved immediately under private `.runs/course/`.

```powershell
gaia-agent run course --runs .runs/course --limit 1
gaia-agent run course --runs .runs/course
gaia-agent export course --runs .runs/course --output .runs/course-payload.json --username YOUR_HF_USERNAME --agent-code https://huggingface.co/spaces/YOUR_HF_USERNAME/YOUR_SPACE/tree/main
gaia-agent submit-course .runs/course-payload.json
```

`run course` fetches questions from the course API and tries `/files/{task_id}` for attachments. If that endpoint fails, an authorized `HF_TOKEN` allows fallback to GAIA validation files. If both fail, the command stops before solving; fix access and run it again. The current course set has two MP3 files, one PNG, one Python file, and one XLSX, plus questions referring to online videos. Do not use `--no-download` for tasks with attachments.

`export course` requires exactly 20 completed, nonempty, short single-line answers and normalizes a public Space link to `/tree/main`. `submit-course` **posts to the course server and updates the student leaderboard**; inspect the local payload before running it. `score-validation course --runs .runs/course` gives an **estimated score** using local exact matching for diagnosis. It requires access to gated GAIA validation data, and its report must remain private. The course API response is authoritative for the score and certificate eligibility. Claim a passing certificate at the [certificate Space](https://huggingface.co/spaces/agents-course/Unit4-Final-Certificate).

## Official GAIA: 301 questions

Accept the [GAIA dataset terms](https://huggingface.co/datasets/gaia-benchmark/GAIA) and freeze the code and configuration. First run a stratified 20-question pilot on validation, where reference answers are available. Proceed to official test only when all 20 questions finish and the local approximation of the official scorer reaches **at least 40%**. The leaderboard accepts test only; local test accuracy is unavailable because its answers are private.

```powershell
gaia-agent preflight validation --runs .runs/validation-pilot
gaia-agent run validation --runs .runs/validation-pilot --manifest .runs/validation-pilot/.meta/pilot-20.json
gaia-agent status validation --runs .runs/validation-pilot
gaia-agent score-validation validation --runs .runs/validation-pilot --manifest .runs/validation-pilot/.meta/pilot-20.json --output .runs/validation-pilot/.meta/score.json

# Continue only when score.json says pilot_complete=true and meets_40_percent_gate=true:
gaia-agent preflight test --runs .runs/test-official
gaia-agent run test --runs .runs/test-official
gaia-agent status test --runs .runs/test-official
gaia-agent export test --runs .runs/test-official --output .runs/gaia-test.jsonl
gaia-agent check-official .runs/gaia-test.jsonl
```

`preflight` makes no cloud model calls. It checks gated dataset access, attachments, local dependencies, and one real audio decode path. It writes `.meta/run-snapshot.json`, `.meta/preflight.json`, and a deterministic 20-question validation manifest with **6/10/4** questions at Levels 1/2/3 and available media types. A `degraded` search-health report does not stop the pilot, but affected answers retain evidence limitations. `score-validation ... --manifest` scores only those 20; incomplete questions cannot pass. If the score is below 40%, inspect the report and failed validation tasks, fix the pipeline, and start a **new** directory. Do not tune on private test questions.

Test preflight checks the full 301-question manifest and all 71 attachments, then freezes a full-run snapshot. Resume test in that same directory. Per-task stage records are written immediately and completed questions are skipped. `.meta/result.json` reports status counts, unfinished task IDs, evidence issues, and token usage for completed records without exposing questions or answers.

`export test` requires 301 completed, nonempty, single-line answers matching the run snapshot and current inputs. Failed reviews, unresolved adjudications, and evidence warnings such as truncation or sampling block export. Inspect affected task records and original sources, then retry recoverable failures with `gaia-agent run test --runs .runs/test-official --task-id TASK_ID`. When you have independently checked the original evidence and must retain an answer, provide a per-task `--review-waivers .runs/review-waivers.json` file. Each entry must match the current `run_signature`, `input_fingerprint`, `evidence_fingerprint`, `answer`, exact `allow` issue list, and a specific `reason`. Global `--accept-partial-evidence` and blank-answer `--fill-failures` cannot produce a strict official submission. `check-official` validates UTF-8 JSONL, unique task IDs, full coverage, and the 93/159/49 level counts against gated test metadata.

After the check passes, complete Hugging Face OAuth within the [official leaderboard Space](https://huggingface.co/spaces/gaia-benchmark/leaderboard), upload `.runs/gaia-test.jsonl`, and provide the agent name, model family, system prompt example, project URL, organization, and contact email. Being signed in to the main site does not complete the Space OAuth step. The current [leaderboard source](https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/app.py) requires an account at least 60 days old and limits each account to one submission per day. It stores the uploaded file and submission date before parsing the JSONL, so an invalid file may use that day's attempt. Scores may take hours to appear. The CLI creates and checks the file; it **does not submit to the official leaderboard**. Do not place the JSONL in a public Space code repository.

## Solver pipeline

Each question has an isolated `task_id` record with its answer, evidence, citations, usage, stage state, shared budget, and task-local memory. Local preparation extracts documents and tables, transcribes audio, and exposes video evidence. DeepSeek V4 Pro can call `inspect_visual` on an image, PDF page, video timestamp, or selected region; Flash processes only requested visual evidence. `query_attachment` runs bounded read-only SQL over CSV/TSV/XLSX, including cell styles, while `calculate` evaluates bounded numeric expressions without arbitrary code execution. The host records search/fetch provenance and marks weak coverage as degraded. In `adaptive` mode, a blind Claude Opus 5.5 reviewer checks Level 2/3, attachment, low-confidence, and weak-evidence tasks; Pro adjudicates disagreements. Memory is isolated to one task, stage checkpoints support resume, and the shared budget limits total work across roles. The optional OpenAI Responses route retains built-in web search and adds Code Interpreter only when `code_interpreter_enabled = true`. Submission files contain only concise answers, not task memory or reasoning records.

**Material processing limits:** Attachments are parsed locally. On-demand image descriptions/OCR can miss fine print, table structure, or brief visual events; a Flash observation is not a complete substitute for the original source. Oversized XLSX text is sampled with a warning; use read-only SQL against the original table for exact counts or filters. Each task prepares at most 24 images; video samples up to 24 frames across the clip and can inspect a requested timestamp again. Sampling and omitted pages/images produce per-task warnings. Table queries, calculations, and visual inspection also have size and call limits; errors or truncation remain visible for review.

Adversarial review is heuristic and cannot guarantee correctness. Check dates, units, list order, missed video frames, and conflicting sources. Use `score-validation` to find pipeline defects, not to memorize validation answers or hard-code answer keys.

## Public Space release allowlist

The course `agent_code` must point to a **public Space with inspectable code** under your account. The `space/` directory supplies a manual question-and-attachment demo. Build the release directory from the project root:

```powershell
python scripts/build_space.py --output dist/space_bundle
```

The allowlist builder places `README.md`, `app.py`, `requirements.txt`, `gaia_agent/`, `pyproject.toml`, and the credential-free `config.example.toml` at its output root. Inspect it, then upload the **contents of the output directory** to the Space repository root. The output path must not exist before building; use a new `--output` path for another build. Verify that the deployed page works. `export course` checks only URL syntax, not whether the Space is public or operational.

Set the required `DEEPSEEK_API_KEY` and `SPACE_ACCESS_TOKEN` as Space Secrets. The default `adaptive` review also needs `ANTHROPIC_API_KEY`. Optionally set `TAVILY_API_KEY` for Tavily search; otherwise the host uses Bing RSS. Set `DEEPSEEK_BASE_URL` and `ANTHROPIC_BASE_URL` only when using non-default endpoints. An OpenAI solver instead needs `OPENAI_API_KEY` and, optionally, `OPENAI_BASE_URL`. `HF_TOKEN` is **optional for this manual demo**; configure it only when gated dataset access or course attachment fallback is needed. The page requires the access token in a password field before running the agent, preventing unrestricted API usage; share it only with intended testers. It accepts manually entered questions and attachments and does not host GAIA questions.

Publish from an **allowlist**: the files above and other reviewed generic documentation only. Exclude `config.toml`, `.env`, `.runs/`, course payloads, test JSONL, Parquet/attachments, caches, logs, and local transcripts. Do not embed benchmark questions or answers in public code. Inspect every uploaded file even when local `.gitignore` excludes private paths.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `GAIA is gated` | Sign in with the same Hugging Face account, accept the dataset terms, and set `HF_TOKEN` or run `hf auth login`. |
| Course attachment 404 or missing | Configure an `HF_TOKEN` with dataset access for validation fallback, then rerun the command. |
| `Install the video extra` / missing transcription package | Install `".[video]"` / `".[audio]"`. Local transcription runs on CPU and may take time to load the model. |
| Model or tool request fails | Use `doctor`. The DeepSeek model IDs are `deepseek-v4-pro` and `deepseek-flash`; Pro cannot directly accept images. Check credentials, quota, shared budget, and the task record. |
| Search preflight is `degraded` | Public backends did not provide enough independent relevant sources. The validation pilot can proceed, but search-dependent answers retain an evidence warning. Configure a reliable search service and start a new run directory when retrying. |
| `Run settings changed` / `Run snapshot differs` | Check code, config, task manifest, and attachments; use a fresh test `--runs` directory and rerun preflight. `--force` only recomputes tasks within the current snapshot. |
| Export or official check fails | Complete failed tasks; check the 20/301 counts, submission fields, and answer format. `check-official` requires gated test metadata. |

## References

- [Agents Course](https://huggingface.co/learn/agents-course/zh-CN/unit0/introduction), [Unit 4 hands-on](https://huggingface.co/learn/agents-course/zh-CN/unit4/hands-on), [course starter template](https://huggingface.co/spaces/agents-course/Final_Assignment_Template)
- [GAIA dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA), [official submission instructions](https://huggingface.co/spaces/gaia-benchmark/leaderboard), [public scorer](https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/scorer.py)
- [DeepSeek models and pricing](https://api-docs.deepseek.com/quick_start/pricing/), [Responses API tool compatibility](https://api-docs.deepseek.com/guides/responses_api/)
