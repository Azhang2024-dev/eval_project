# kb-qa-eval

知识库问答质量评测工具。给它一张包含「用户问题 + 模型回答 + 检索到的知识库片段」的表格（**最好是 pipeline 过后的 zhfmt 结尾格式的文件**），它调用裁判模型给每条回答打分：5 个维度、1-5 分、每个分数都附打分理由，产出带评分理由的原始评测工作簿（统计报告由 kb-report 生成）。

**强烈建议使用 pipeline 之后的 zhfmt 文件进行评测，以尽量减小最终的分数误差**

两种用法：

- **当 CLI 用**：直接跑 Python 脚本，适合批处理、挂任务、接进流水线。
- **当 Agent Skill 用**：放进 agent 的 skills 目录，对 agent 说"帮我评测这份问答"，由 agent 代为收集参数、展示确认、执行。

两种用法跑的是同一套代码，评分逻辑完全一样。

## 评分方式

5 个维度分别打 1-5 分，再加权得到总分：

| 维度 | 权重 | 看什么 |
|------|------|--------|
| 回答相关性 | 30% | 有没有答到问题点上 |
| 引用准确性 | 30% | 是否基于检索到的知识库内容，有没有编造、错引 |
| 范围完整性 | 15% | 问题涉及的点是否覆盖全了 |
| 指令遵循 | 15% | 是否遵守问题要求的格式、结构、约束 |
| 阅读感受与自然度 | 10% | 读起来是否清晰、自然 |

加权均分 = 相关性×0.3 + 引用×0.3 + 完整性×0.15 + 指令×0.15 + 自然度×0.1

验收线：

- MOS（加权均分）≥ 4.1 算达标
- 加权均分 < 3.0 的条目算 badcase，badcase 率应 < 10%

两条特殊规则：

- 回答为空 → 五个维度直接全 1 分，理由写"回答为空"（不发给裁判模型，模型给空回答打分没有意义）。
- 检索缺失（来源文件名、搜索结果都拿不到）→ 同样本地规则判 1 分，控制台统计里单独计数。这类条目通常是上游检索环节出了问题，统计里标出来方便排查。

完整评分标准和 few-shot 示例见 `references/eval_standard.md`。

## 安装

Python 3.9+（我们在 3.12 上开发和测试）。

```bash
pip install -r requirements.txt
```

依赖只有五个：pandas、openpyxl、requests、pymupdf（复核阶段读取 PDF 原文用）、fast-langdetect（语言识别，首次使用自动下载模型）。

评测要调模型 API，密钥通过环境变量提供，打开powershell或者bash，输入以下命令：

```powershell
# PowerShell
$env:KB_QA_API_KEY = "你的API密钥"
$env:KB_QA_API_BASE_URL = "https://你的网关地址/v1"   # 可选，不设就用默认地址
```

```bash（命令符cmd）
# bash
export KB_QA_API_KEY="你的API密钥"
export KB_QA_API_BASE_URL="https://你的网关地址/v1"
```

密钥只从环境变量读取。不要把它写进代码、JSON 配置或提交到仓库——程序也不会把它写进 Excel、checkpoint 或 plan 文件里。

跑 `--plan`（预检查，见下文）不需要密钥，因为这一步不调 API。

## 输入文件要求

支持 `.xlsx` / `.csv` / `.json` / `.jsonl`。CSV 优先按 UTF-8 读取，失败自动回退 GB18030（兼容老的中文编码文件），plan 里会显示实际使用的编码。

列名不用和下面完全一致，程序按关键词自动识别（大小写不敏感）：

| 字段 | 能识别的列名举例 | 说明 |
|------|------------------|------|
| 问题 | query、问题、question | 必需 |
| 检索片段 | 搜索结果、chunk、检索、retrieved、context | 必需 |
| 回答 | answer、回答、response、大模型问答结果 | |
| 来源文件名 | filename、文件名、source、来源 | 多文件来源时会逐行收集去重 |
| 标注答案 | 标注、reference、gold、标准答案 | 仅留给人看，不进评分 prompt |
| 片段分数 | score、分数、相似度、similarity | 检索相似度，可选 |

两种数据布局都支持：

- **布局 A（逐行）**：一行就是一条完整问答（问题、回答、片段各一列）。每行独立评测；相同 query 的多行**不会合并**，各评各的。query 为空的整行跳过不评测（plan 里会提示跳过的行数）。
- **布局 B（分组）**：一个问题占多行、每行一个检索片段（问题列只在第一行有值，后续行为空）。程序按问题分组，把同一问题的片段合并成一条再评；组内答案取第一个非空值。

布局自动检测按"query 列为空的行占比 >30%"判断，拿不准时用 `--layout row_by_row|grouped` 手动指定（plan 里会记录 `layout_source: auto|manual` 供审计）。

关于标注答案列：它**不会**喂给裁判模型。评分只看「问题 + 检索片段 + 回答」，标注答案留在结果表里供人工比对。不然裁判模型会变成对着标准答案比对，打出来的分不可信。

## 用法一：直接跑 CLI

标准流程分两步：**先生成 plan，确认后再执行**。

这么设计是因为每条评测都要花钱调 API。程序要求先确认列映射和参数识别无误，才允许真正开跑。直接只传 `--file` 和 `--models` 会被拒绝。

### 第 1 步：生成 plan

```powershell
python scripts/evaluate.py --file D:\data\input.xlsx --models "azure/gpt-5.4,bytedance/deepseek-v4-pro" --plan --plan-output plan.json
```

这一步做本地预检查：文件能不能读、格式是什么、列识别成了哪几个字段、一共多少条 query、每个模型用什么协议调用。有问题直接报错，没问题就把检查结果写进 `plan.json`。

先看看 plan 内容，确认列映射和条数符合预期，再往下走。

### 第 2 步：执行

```powershell
python scripts/evaluate.py --execute-plan plan.json
```

执行时会重新校验输入文件指纹（防止 plan 之后你把文件换了），然后默认 10 路并发评测，进度实时打印。每个模型各出一份结果，文件名带模型全名：

```
input_evaluated_azure_gpt-5.4.xlsx
input_evaluated_bytedance_deepseek-v4-pro.xlsx
```

文件名带 provider 前缀是有意的：不同厂商可能有同名短模型，带上 provider 才不会互相覆盖。

### 中断续跑

被打断了（网络断、Ctrl+C、机器重启）直接重跑同一条命令即可。checkpoint 文件记在输入文件旁边（`_checkpoint_<模型>_<协议>.json`），已经跑完的条目跳过，只补剩下的。

如果改了 prompt 或评分逻辑，旧 checkpoint 会按版本门禁自动作废，不会拿旧口径的结果混数。checkpoint 还会校验 `response_mode` 与结构版本：换了协议/响应模式或结构升级后，旧 checkpoint 一律拒绝续传，重新生成 plan 即可。

### 用配置文件传参

参数多的时候可以写成 JSON（参考 `examples/eval_config.example.json`）：

```json
{
  "file": "D:/data/input.xlsx",
  "language": "en",
  "models": ["azure/gpt-5.4", "bytedance/deepseek-v4-pro"],
  "model_protocols": {
    "azure/gpt-5.4": "chat_completions",
    "bytedance/deepseek-v4-pro": "chat_completions"
  },
  "workers": 10,
  "thinking": true,
  "thinking_effort": "high",
  "review": false,
  "merge": false
}
```

```powershell
python scripts/evaluate.py --config config.json --plan --plan-output plan.json
python scripts/evaluate.py --execute-plan plan.json
```

命令行上显式传的参数优先于配置文件。

### 可选：低分复核

生成 plan 时带上 `--review`，执行过程中会对低分条目做跨模型复核：**任一维度打 1 分（该维度核心任务失败），或加权均分低于 badcase 线（默认 3.0）**的条目，交给其他模型重新审一遍。

```powershell
python scripts/evaluate.py --file D:\data\input.xlsx `
  --models "azure/gpt-5.4,alibaba/qwen3.7-max,bytedance/deepseek-v4-pro" `
  --review --doc-dirs D:\kb\docs `
  --review-models "azure/gpt-5.4,alibaba/qwen3.7-max,bytedance/deepseek-v4-pro" `
  --threshold 3.0 `
  --plan --plan-output plan.json
```

说明：

- `--doc-dirs` 是知识库原文目录。复核低分引用要回去翻原文，所以必须提供。`.md`/`.txt`/`.pdf`/`.docx` 都能直接读取原文（扫描版 PDF 无文本层时会注明）；其他格式仍以占位符送审。
- 复核会排除正在被评的那个模型（避免自评）。比如三个模型互评时，每个模型的低分条目由另外两个复核。
- 复核结果单独写在输出工作簿的「低分复核」sheet 里，不改原评分。每个复核模型给出：结论（合理/需调整）、建议分（每维度）、主要问题（含原文对照证据）、备注。建议分仅供人工参考，不会自动改写正式评分。
- 复核有独立的断点续传：中断后重跑会跳过已复核的条目（文档库内容、复核模型或复核 prompt 变化时自动作废重来）。

### 可选：多模型结果合并

几个模型各评完一份之后，想做横向对比：

```powershell
python scripts/merge.py `
  --file "azure/gpt-5.4=D:\data\input_evaluated_azure_gpt-5.4.xlsx" `
  --file "bytedance/deepseek-v4-pro=D:\data\input_evaluated_bytedance_deepseek-v4-pro.xlsx" `
  --output merged.xlsx
```

合并时做严格校验：语言、原始列、query 内容、行序、输入指纹必须一致，不匹配直接拒绝——两份表可能根本不是同一批数据，硬合并只会给出错误的对比。没有 metadata 的旧结果可以合并，但会带警告。

合并工作簿有三个 sheet：逐条模型对比（宽表：原始字段 + 各模型分数理由）、模型统计、数据校验。

也可以在评测时直接带 `--merge` / `--merge-output`，跑完各模型后顺手生成。

### 参数一览（evaluate.py）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--config <path>` | 无 | JSON 配置文件；命令行显式参数优先 |
| `--plan` | off | 只做预检查并输出 plan，不调 API |
| `--plan-output <path>` | 无 | plan JSON 写到哪 |
| `--execute-plan <path>` | — | 按指定 plan 执行，不接受其余覆盖参数 |
| `--file <path>` | 必填 | 输入文件 |
| `--language` | auto | 评测语言 zh/en/fr，缺省自动检测 |
| `--layout` | auto | 数据布局覆盖：auto / row_by_row / grouped |
| `--max-chunks N` | 0 | 每 query 只取分数最高的前 N 个 chunk（0=不限，默认关闭） |
| `--max-chunk-chars N` | 0 | 单个 chunk 截断到 N 字符（0=不限） |
| `--output <path>` | 自动 | 输出文件名基准，模型后缀自动追加 |
| `--test N` | 无 | 只评前 N 条，试跑用 |
| `--workers N` | 10 | 并发数 |
| `--checkpoint-dir <dir>` | 输入文件所在目录 | checkpoint 目录 |
| `--thinking` / `--no-thinking` | off | 开关 thinking 模式 |
| `--thinking-effort` | high | low/medium/high/max，需配合 `--thinking` |
| `--models` | 必填 | 逗号分隔的模型列表 |
| `--model-protocols` | 内置映射 | `MODEL=PROTOCOL`，协议：chat_completions / responses / anthropic_messages |
| `--model-response-modes` | text | `MODEL=MODE`，mode：text / json_schema（后者仅 Responses 协议可用） |
| `--review` / `--no-review` | off | 开关低分复核 |
| `--doc-dirs` | review 必填 | 知识库原文目录，逗号分隔 |
| `--review-models` | review 必填 | 复核模型列表 |
| `--threshold N` | 3.0 | 复核触发阈值（加权均分 badcase 线）；任一维度 1 分恒触发 |
| `--merge` / `--no-merge` | off | 开关多模型对比表 |
| `--merge-output <path>` | 自动 | 对比工作簿路径 |

参数一览（merge.py）：

| 参数 | 说明 |
|------|------|
| `--file MODEL=PATH` | 各模型结果文件，可重复传 |
| `--output` | 合并输出路径，默认生成在第一个输入文件旁边 |
| `--language` | 预期语言，默认自动从表头识别 |
| `--model-protocols` / `--model-response-modes` | 同上 |

## 用法二：作为 Agent Skill

如果你平时用 opencode、Claude Code 这类能读 `SKILL.md` 的 agent，上面的命令都不用手敲。

### 安装

整个目录放进 agent 的 skills 目录即可，例如：

```
~/.agents/skills/kb-qa-eval/
```

（Windows 下对应 `C:\Users\<用户名>\.agents\skills\kb-qa-eval`。）环境变量照样要设。

### 怎么用

直接说话，把文件路径和模型告诉它：

> 帮我评测 D:\Downloads\en_llm_sampled_100.xlsx，用 bytedance/deepseek-v4-pro 打分

agent 接下来会：

1. 缺什么参数就问什么（语言、要不要复核、原文目录在哪），不会自作主张；
2. 跑 `--plan`，把识别结果贴给你看——列映射到了哪些字段、多少条 query、每个模型走什么协议；
3. 你确认后它才执行，进度实时可见；
4. 跑完给你输出文件路径和控制台统计；中途断了重发一句"继续"，它从 checkpoint 接着跑。

整个过程 headless：不弹窗、不在终端里等输入，所有交互都在聊天窗口里。这是这个 skill 的硬性要求，方便无人值守挂着跑（比如睡前发起，早上看结果）。

### CLI 和 Skill 怎么选

- 想脚本化、定时跑、接 CI → CLI。
- 在对话里干活，想省掉拼参数和确认流程 → Skill。

能力完全对等，选哪个只看你在哪干活。

## 输出文件长什么样

每个模型一份 xlsx，含两个 sheet：

**Evaluation**（主表）：

- A-F 列：原始列原样保留（问题、标注、文件名、搜索结果、片段分数、回答）
- G-P 列：5 个维度的分数和理由，各占两列
- Q 列：加权均分，由本地 Python 按公式算出——不是裁判模型自己报多少就算多少
- R 列：综合评价（高分原因、低分原因、改进建议）

**统计报告**（统计摘要 + Badcase 归因分析）由独立的 kb-report skill 生成，格式见其说明；评测工作簿本身只含上面的 Evaluation 表。控制台统计输出固定中文。

开了复核的话还有第三个 sheet **低分复核**（en 下为 Review），记录每条低分条目的复核明细：各复核模型结论、建议分、主要问题与备注。

## 跑测试

```powershell
venv\Scripts\python.exe -m unittest discover -s tests
```

（没建虚拟环境的话，用你的 python 跑后面的 `python -m unittest discover -s tests` 即可。）

测试里包含等价回归测试：拿生成结果和 golden 基准对比，改代码时防止悄悄改变行为。

## 目录结构

```
kb-qa-eval/
├── SKILL.md                  # Skill 入口：agent 读这个文件学会用法
├── README.md                 # 本文件
├── requirements.txt
├── scripts/
│   ├── evaluate.py           # 主入口（plan / execute / review）
│   ├── merge.py              # 多模型结果合并入口
│   ├── file_detector.py      # 格式/列名/布局自动识别
│   ├── prompt_builder.py     # 评测 prompt 组装（标准 + few-shot + 数据）
│   ├── api_client.py         # 并发调用、重试、JSON 解析
│   ├── excel_writer.py       # 结果写 Excel
│   ├── eval_stats.py         # 统计口径唯一源（权重/分布/badcase）
│   ├── eval_config.py        # 配置加载与校验
│   ├── review_module.py      # 低分跨模型复核
│   ├── result_merger.py      # 多模型合并核心逻辑
│   └── report_generator.py   # 控制台统计
├── prompts/                  # 中/英/法 prompt 模板
├── references/
│   └── eval_standard.md      # 评测标准全文 + few-shot 示例
├── examples/
│   └── eval_config.example.json
└── tests/                    # 单元测试 + 等价回归测试
```
