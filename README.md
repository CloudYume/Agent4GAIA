# GAIA Course Agent

[English documentation](README_EN.md)

一个可续跑的 GAIA 解题 CLI。它用同一套解题与复核流程支持两条**不同的**评测链路；课程成绩不能直接作为正式 GAIA 排行榜成绩提交。

| 链路 | 题目 | 提交物 | 去向 |
| --- | --- | --- | --- |
| Hugging Face Agents Course Unit 4 | GAIA 验证集 Level 1 中筛选的 20 题 | `username`、公开 Space 代码链接、20 个 `submitted_answer` | [课程评分 API](https://agents-course-unit4-scoring.hf.space/docs)及[学生榜](https://huggingface.co/spaces/agents-course/Students_leaderboard)；课程介绍写 30% 门槛，证书页面写高于 30%，建议至少达到 35% |
| GAIA 正式榜 | 2023 test 的 301 题：Level 1/2/3 分别为 93/159/49 | 301 行 JSONL，逐行包含 `task_id` 和 `model_answer` | [GAIA 排行榜](https://huggingface.co/spaces/gaia-benchmark/leaderboard)的上传表单；测试答案不公开 |

GAIA [数据集](https://huggingface.co/datasets/gaia-benchmark/GAIA)需要先登录并接受门控条款。不要把验证/测试题目、附件、答案或运行记录放入公开 Space、Git 仓库或其他可抓取位置。

## 课程内容与实现

| 课程单元 | 本项目对应实现 |
| --- | --- |
| Unit 1：智能体循环与工具 | `gaia_agent/agent.py` 的解题提示、结构化答案及 `deepseek.py` 的函数工具循环，结合逐题记录形成思考、行动、观察闭环。 |
| Unit 2：框架、多智能体与视觉浏览 | `orchestrator.py` 编排 DeepSeek V4 Pro 主解题器、Claude Opus 5.5 盲复核器及 Pro 争议裁决；`memory.py` 保存逐题证据，`contracts.py` 约束共享预算。`attachments.py`、`media.py` 提供媒体定位，模型通过 `inspect_visual` 按需调用 DeepSeek Flash。借鉴课程编排思路，未直接依赖 smolagents。 |
| Unit 3：Agentic RAG | `web_tools.py` 聚合公开检索后端并标记弱覆盖；`sources.py` / `attachments.py` 提取材料，`tools/data.py` 提供只读表格 SQL 与受限数值计算。当前没有独立向量索引。 |
| Unit 4：GAIA 项目与评估 | `cli.py`、`store.py`、`scoring.py`、`validation.py` 负责分层试跑、逐题续跑、输入快照、已知答案验证集评分及正式榜文件校验。 |

## 安装与配置

需要 Python 3.11+。在 PowerShell 中从项目根目录运行：

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[audio,video,dev]"
Copy-Item config.example.toml config.toml
.\.venv\Scripts\gaia-agent doctor
```

`audio` 安装本地 CPU Whisper 转写依赖；默认 `transcription_provider = "local"` 需要它。若设为 `"openai"`，可仅安装 `".[video,dev]"`，并配置转写模型。`video` 安装 `yt-dlp` 和 `imageio-ffmpeg`，用于题目中的 YouTube 链接、视频抽帧与音轨提取；下载仍取决于网络和站点可用性。`[agent] youtube_cookies_from_browser` 默认留空，不读取浏览器 Cookie；确需使用已登录的 Edge Default profile 时填写 `"edge:Default"`。Edge 运行时 Cookie 数据库可能被锁定，关闭浏览器后仍可能遇到应用绑定加密；不要把 Cookie 文件放进代码仓库或日志。`dev` 只用于本地测试。

在私有的 `config.toml` 填入实际可用的模型 ID 与凭证。默认解题和争议裁决使用官方 DeepSeek Chat Completions API 的 `deepseek-v4-pro`；模型请求检查图像或视频帧时，同一 DeepSeek 凭证才调用 `deepseek-flash` 描述/OCR。Pro 本身不支持图像输入。独立复核使用 Anthropic Messages API 的 Claude Opus 5.5，并可使用其服务端网页搜索。DeepSeek 的本地搜索聚合 Bing RSS、Bing HTML、Wikipedia 等公开入口，可选 `TAVILY_API_KEY` 增加 Tavily；无独立相关来源时明确标记降级，而不是把单条 Wikipedia 命中当作充分证据。当前环境的公开搜索入口可能只返回弱结果，预检会报告这一点；试跑仍可继续，但相关题目需要证据复核。DeepSeek 的 Responses API 会忽略内置 `web_search` 与 `code_interpreter`，因此不能只把旧 OpenAI 配置的模型名换成 Pro。`code_interpreter_enabled = false` 是 DeepSeek 默认配置；仅选择 OpenAI Responses 求解器时该开关才控制其内置 Code Interpreter。`review_mode = "off"` 时只需解题服务凭证；`"adaptive"` 或 `"always"` 还需要复核服务凭证。环境变量 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`ANTHROPIC_API_KEY`、`ANTHROPIC_BASE_URL`、`HF_TOKEN` 优先于配置文件；选择 OpenAI 求解器时使用 `OPENAI_API_KEY`、`OPENAI_BASE_URL`。`HF_TOKEN` 用于门控数据集及课程附件的后备下载；课程题目列表本身公开。`doctor` 仅显示配置摘要和密钥是否存在，不会调用模型。

`[agent]` 中的 `search_context_size = "medium"` 只控制可选 OpenAI Responses 内置网页搜索的上下文规模；DeepSeek 搜索由本地工具限制结果数量和文本长度。`max_tool_calls = 12` 对 DeepSeek 限制一次解题或裁决中由本地主机执行的函数工具次数（硬上限 12；设为 `0` 时仍使用内部默认预算 8）；选择 OpenAI Responses 时，正数会传给 API，`0` 则省略该参数。DeepSeek 官方 Responses API 会忽略 `max_tool_calls`，所以 DeepSeek 预算必须由本地循环执行。`max_output_tokens = 8000` 限制每次模型请求的输出 token（包含推理 token）；独立 Anthropic 复核有自己的上限。这些是单题/单次请求预算，并非整个批次的费用硬上限；搜索、Flash 图像预处理、重试、复核和裁决均会增加用量。上限过低可能让模型无法产出完整答案。

主要参数见 [config.example.toml](config.example.toml)。`workers` 控制并发题数，可能影响费用与速率限制；`confidence_threshold` 只用于 `adaptive` 复核。配置、提示词和关键实现源码的哈希共同形成运行指纹。正式 test 批次还在 `.meta/run-snapshot.json` 固定完整题单与附件输入指纹；更改代码、配置或输入后必须使用新的 `--runs` 目录，`--force` 不能绕过快照校验。

以下命令假设已激活虚拟环境；未激活时将 `gaia-agent` 换成 `.\.venv\Scripts\gaia-agent`。需要指定其他配置文件时，把全局参数放在子命令前：`gaia-agent --config my-config.toml doctor`。

## 课程 20 题

先用 `--limit 1` 检查完整解题路径，再续跑剩余题目；已完成的题目默认跳过。每题结果即时写入私有的 `.runs/course/`。

```powershell
gaia-agent run course --runs .runs/course --limit 1
gaia-agent run course --runs .runs/course
gaia-agent export course --runs .runs/course --output .runs/course-payload.json --username YOUR_HF_USERNAME --agent-code https://huggingface.co/spaces/YOUR_HF_USERNAME/YOUR_SPACE/tree/main
gaia-agent submit-course .runs/course-payload.json
```

`run course` 从课程 API 获取题目，通过 `/files/{task_id}` 获取关联文件。如果该接口失败，程序可使用已授权的 `HF_TOKEN` 从 GAIA 验证集回退下载；两处均失败时会在解题前报错，需要排查后重试。课程当前有 2 个 MP3、1 个 PNG、1 个 Python 文件和 1 个 XLSX 附件，另有网页视频题。不要使用 `--no-download` 运行需要附件的题目。

`export course` 要求恰好 20 个已完成、非空、单行的简短答案，并自动规范公开 Space 链接为 `/tree/main`。`submit-course` 是**实际向课程服务器提交并更新学生榜**的写操作，只在检查本地 payload 后运行。`score-validation course --runs .runs/course` 使用本地严格匹配规则给出**估算分数**，用于诊断；它需要授权访问 GAIA 验证集，报告应留在私有目录。课程成绩和证书资格以课程 API 实际返回为准；证书通过[证书页面](https://huggingface.co/spaces/agents-course/Unit4-Final-Certificate)领取。

## GAIA 正式榜 301 题

在 [GAIA 数据集](https://huggingface.co/datasets/gaia-benchmark/GAIA)接受门控后，冻结代码与配置，先在有公开标准答案的 validation 集运行分层 20 题。只有 20 题全部完成、按本地官方近似规则得分至少 **40%**，才进入正式 test。正式榜仅接收 test；本地无法获得私有 test 正确率。

```powershell
gaia-agent preflight validation --runs .runs/validation-pilot
gaia-agent run validation --runs .runs/validation-pilot --manifest .runs/validation-pilot/.meta/pilot-20.json
gaia-agent status validation --runs .runs/validation-pilot
gaia-agent score-validation validation --runs .runs/validation-pilot --manifest .runs/validation-pilot/.meta/pilot-20.json --output .runs/validation-pilot/.meta/score.json

# 仅在 score.json 中 pilot_complete=true 且 meets_40_percent_gate=true 后继续：
gaia-agent preflight test --runs .runs/test-official
gaia-agent run test --runs .runs/test-official
gaia-agent status test --runs .runs/test-official
gaia-agent export test --runs .runs/test-official --output .runs/gaia-test.jsonl
gaia-agent check-official .runs/gaia-test.jsonl
```

`preflight` 不调用云端模型；它检查门控数据集、附件、本地依赖及一个真实音频解码路径，在运行目录写入 `.meta/run-snapshot.json`、`.meta/preflight.json` 及固定的 20 题分层清单。validation 清单按 Level 1/2/3 选取 **6/10/4** 题，并覆盖可用的媒体类型；它用于可评分链路试跑，绝不可上传正式榜。搜索健康状态可能为 `degraded`，这不会阻止试跑，但相关题目会留下证据降级标记。`score-validation ... --manifest` 只评分这 20 题；未完成题不算通过。若低于 40%，检查报告与失败题，在 validation 上修复后用**新**目录重新预检，不要用 test 题调参。

test 预检会核对完整 301 题与 71 个附件，并固定全量快照；之后只在同一目录续跑。每题阶段结果即时保存，重启时跳过已完成题。`.meta/result.json` 给出状态数量、失败题号、证据问题及已完成记录的 token 用量，不包含题目正文或答案。

`export test` 只接受与快照及当前输入匹配的 301 个已完成、非空、单行答案。复核失败、未裁决争议和材料截断/采样等证据问题均会阻止导出。先检查对应的逐题 JSON 与原始材料，再对可恢复问题重跑，例如 `gaia-agent run test --runs .runs/test-official --task-id TASK_ID`。确实核实过原始证据且需要保留该答案时，可提供逐题 `--review-waivers .runs/review-waivers.json`；每项必须绑定当前 `run_signature`、`input_fingerprint`、`evidence_fingerprint`、`answer`、准确的 `allow` 问题列表和具体 `reason`。全局 `--accept-partial-evidence` 与空答案 `--fill-failures` 均不能生成严格的正式提交。`check-official` 从门控 test 元数据核验 UTF-8 JSONL、唯一题号、完整覆盖及 93/159/49 的等级数量。

通过检查后，在[正式榜](https://huggingface.co/spaces/gaia-benchmark/leaderboard)完成 Space 的 Hugging Face OAuth 登录并上传 `.runs/gaia-test.jsonl`，同时填写 Agent 名称、模型家族、系统提示示例、项目 URL、组织和联系邮箱。主站已登录不代表 Space OAuth 已完成。当前[榜单源码](https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/app.py)限制账户注册满 60 天、同一账户每天最多提交一次，并可能在数小时后才显示成绩；服务端在解析文件前记录上传和提交日期，所以格式错误也可能耗掉当天机会。CLI 只生成和核验文件，**不会代替你提交正式榜**。不要把 JSONL 上传到公开 Space 的代码仓库。

## 解题链路

每题按 `task_id` 独立保存答案、证据、引用、调用量、阶段状态、共享预算和逐题记忆。本地先准备文档/表格、音频转写与视频素材，向解题器提供可定位的材料。DeepSeek V4 Pro 可按需调用 `inspect_visual` 检查原图、PDF 页面、视频时间点或局部区域；Flash 只处理被请求的视觉内容。`query_attachment` 对 CSV/TSV/XLSX 执行受限只读 SQL，可检查行列与样式；`calculate` 执行受限数值表达式，不运行任意代码。主机执行网页搜索与抓取并记录来源质量；没有相关独立来源时标记证据降级。`adaptive` 模式让 Claude Opus 5.5 对 Level 2/3、附件、低置信度或弱证据题目进行盲复核；分歧交由 Pro 裁决。逐题记忆仅服务当前题目，阶段检查点支持中断后续跑，共享预算约束整题的模型调用和用量。可选 OpenAI Responses 路径继续使用其内置网页搜索，并仅在 `code_interpreter_enabled = true` 时附加 Code Interpreter。最终提交物只保留简短答案，不上传逐题记忆或推理记录。

**材料处理限制：**附件在本地解析；按需视觉描述/OCR 可能遗漏小字、表格结构或瞬时画面，不应把 Flash 观察当作原文件的完整替代。XLSX 内容超出文本预算时按行采样并留下告警；需要精确统计时优先用只读 SQL 查询原始表格。每题最多准备 24 张图像；视频全片均匀抽帧最多 24 帧，并允许按时间点再次检查。抽帧和省略页/图像会在逐题记录中告警；瞬时事件仍可能落在采样间隔内。表格查询、计算及视觉检查均有大小与调用限制，失败或结果截断会留下可审查痕迹。

这是启发式多智能体复核，并不保证答案正确。检查易错的日期、单位、列表顺序、视频帧遗漏及来源冲突。`score-validation` 仅用于发现流程缺陷，不应通过搜索公开答案或把验证题答案写入规则来优化成绩。

## 公开 Space 发布白名单

课程提交的 `agent_code` 必须指向你名下**公开、可查看代码**的 Space。`space/` 提供手动输入问题与附件的演示入口。从项目根目录构建发布目录：

```powershell
python scripts/build_space.py --output dist/space_bundle
```

脚本按白名单复制文件：构建产物根目录包含 `README.md`、`app.py`、`requirements.txt`、`gaia_agent/`、`pyproject.toml` 和无密钥的 `config.example.toml`。先检查产物，再将**产物目录内容**上传到 Space 仓库根目录。目标目录必须预先不存在；再次构建时指定新的 `--output` 路径。部署后实际确认页面可运行。`export course` 仅检查 URL 格式，不能验证 Space 的公开性或运行状态。

在 Space Secrets 设置必需的 `DEEPSEEK_API_KEY` 和 `SPACE_ACCESS_TOKEN`。默认 `adaptive` 复核还需要 `ANTHROPIC_API_KEY`；可选 `TAVILY_API_KEY` 以使用 Tavily 搜索，否则使用 Bing RSS。使用非默认端点时按实际配置设置 `DEEPSEEK_BASE_URL`、`ANTHROPIC_BASE_URL`。改用 OpenAI 求解器时才需要 `OPENAI_API_KEY` 及可选的 `OPENAI_BASE_URL`。`HF_TOKEN` 对这个手动演示页面**可选**，只有需要访问门控数据集或课程附件后备下载时才配置。页面要求先在密码框输入访问令牌才会调用智能体，避免公开页面被任意使用造成 API 费用；只向预期测试者提供该令牌。演示页面只接受手动输入，不托管 GAIA 题目。

建议采用**白名单**上传：上述文件及其他经过检查的通用文档。不要上传 `config.toml`、`.env`、`.runs/`、课程 payload、测试 JSONL、Parquet/附件、缓存、日志或本地转写文本。公开代码也不应内嵌题目或答案。上传前逐文件检查，即使本地 `.gitignore` 已忽略敏感路径也一样。

## 常见故障

| 现象 | 检查 |
| --- | --- |
| `GAIA is gated` | 登录同一 Hugging Face 账户并接受数据集条款；设置 `HF_TOKEN` 或运行 `hf auth login`。 |
| 课程附件 404 或缺失 | 配置有数据集访问权限的 `HF_TOKEN` 以使用验证集后备下载，随后重新运行。 |
| `Install the video extra` / 找不到转写包 | 安装 `".[video]"` / `".[audio]"`；本地转写使用 CPU，首次载入模型需要时间。 |
| 模型或工具请求失败 | 用 `doctor` 检查配置；DeepSeek 模型 ID 为 `deepseek-v4-pro` 与 `deepseek-flash`，Pro 不支持直接视觉输入。核对凭证、额度、共享预算和逐题记录。 |
| 搜索预检 `degraded` | 当前公开检索入口没有足够的独立相关来源；验证集试跑仍可继续，但搜索依赖题目会标记证据降级。可配置可靠检索服务后用新运行目录重试。 |
| `Run settings changed` / `Run snapshot differs` | 核对代码、配置、题单与附件；正式 test 使用新 `--runs` 目录重新预检。`--force` 只用于当前快照内重算。 |
| 无法导出或正式榜核验失败 | 补齐失败题；核对 20/301 题数量、提交字段与答案格式；`check-official` 需要门控 test 元数据。 |

## 参考

- [Agents Course 全部单元](https://huggingface.co/learn/agents-course/zh-CN/unit0/introduction)、[Unit 4 动手实践](https://huggingface.co/learn/agents-course/zh-CN/unit4/hands-on)、[课程模板](https://huggingface.co/spaces/agents-course/Final_Assignment_Template)
- [GAIA 数据集](https://huggingface.co/datasets/gaia-benchmark/GAIA)、[正式榜提交说明](https://huggingface.co/spaces/gaia-benchmark/leaderboard)、[公开评分器](https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/scorer.py)
- [DeepSeek 模型与价格](https://api-docs.deepseek.com/quick_start/pricing/)、[Responses API 工具兼容性](https://api-docs.deepseek.com/guides/responses_api/)
