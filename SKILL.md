---
name: kb-qa-eval
description: "Use this skill for knowledge base QA evaluation (知识库问答评测, KB QA evaluation) — scoring model answers on 5 dimensions (相关性, 引用准确性, 范围完整性, 指令遵循, 阅读感受与自然度) with a 1-5 scale, configurable model protocols, and structured Excel output. Trigger when the user asks to evaluate/score/评测 knowledge base Q&A quality, runs evaluation, checks MOS scores, or reviews answer quality against retrieved context."
---

# KB QA Evaluation (kb-qa-eval)

## Overview

知识库问答质量评测：对大模型回答基于召回知识库内容进行 5 维度 1-5 分自动评分，产出带评分理由的原始评测工作簿。

**本 skill 只做评测**（输入 query 文件 → 各模型原始评测工作簿）。统计摘要与 Badcase 归因报告由 [kb-report](~/.agents/skills/kb-report/) 生成。

## When to Use

- 用户要求评测知识库问答系统的回答质量（5 维度打分：相关性/引用/完整性/指令/自然度）
- 批量评测 Excel/xlsx/csv/json/jsonl 格式的问答数据
- 需要多模型并行评测、断点续传、低分复核、多模型合并

**Do NOT use when:** 需要统计报告/归因分析（用 kb-report）、query 筛选抽样（用 kb-query-pipeline）、主观人工逐条审核、评测目标不是知识库问答。

## Input → Output Contract

**Input**: 含 query + 回答 + 召回片段的表格（列名自动识别，布局 A 逐行 / B 按 query 合并自动判断）

**Output**: 每模型一个 `<file>_evaluated_<provider>_<model>.xlsx`，含两个 sheet：
- `评测结果`：原数据列 + 5 维度×（分数/理由）+ 加权均分 + 综合评价
- `低分复核`（`--review` 开启时）：低分条目跨模型复核结果

报告（统计摘要/归因）不在本 skill 输出内。

## Standard Flow

```bash
python scripts/evaluate.py --config config.json --plan --plan-output plan.json
# 聊天框展示 plan manifest，用户确认
python scripts/evaluate.py --execute-plan plan.json
```

- 禁止绕过 plan 直接执行（会被拒绝）；headless，无弹窗、无 stdin 等待
- 配置示例：`examples/eval_config.example.json`（评测模型/协议/复核等 profile 都在 config 里）；完整参数见 `python scripts/evaluate.py --help`

## Languages

| 数据语言 | 检测 | 评测标准 |
|---|---|---|
| zh（中文） | 汉字脚本 | 中文 prompt（zh） |
| en/fr/ar/id/ru（及其他非中文） | fast-langdetect（fr/id/en）、脚本特征（ar/ru） | 统一英文 prompt |

非中文数据一律用英文评测标准，评分理由输出英文；检测到的真实语种仅用于展示。

## Runtime API Configuration

```powershell
$env:KB_QA_API_KEY = "<your-api-key>"          # 正式执行必填
$env:KB_QA_API_BASE_URL = "https://maragate.aibotplatform.com/v1"  # 可选
```

`--plan` 不需要 API Key。**禁止**把 API Key 写入脚本/配置/版本库。

## Scoring Dimensions

5 维度加权打分（相关性 30% / 引用准确性 30% / 范围完整性 15% / 指令遵循 15% / 阅读感受与自然度 10%），MOS 准出 ≥4.1，Badcase <3.0 占比 <10%，回答为空全部维度 1 分。完整维度定义、验收线与 few-shot 见 [README#评分方式](README.md#评分方式) 与 [references/eval_standard.md](references/eval_standard.md)。

## Common Mistakes

- **未确认列映射就运行**: 先 `--plan` 并在聊天中展示确认
- **忽略空 query / 未识别 answer 列**: plan 的 `warnings` 会提示，确认时注意
- **布局 B 未合并 chunks**: 同一 query 多个召回片段必须合并评测，布局自动判断
- **多文件召回未带归属**: 引用准确性以全部召回内容为准，禁止单文件名锚定

## Multi-model Merge

```bash
python scripts/merge.py --file MODEL=PATH --file MODEL=PATH
```

合并工作簿含 `逐条模型对比`/`模型统计`/`数据校验`；严格匹配语言、原列、query 身份、行序与输入指纹。

## References

- 评分标准: [references/eval_standard.md](references/eval_standard.md)
- 报告生成: `~/.agents/skills/kb-report/`
- 测试: `venv\Scripts\python.exe -m unittest discover -s tests`（含 golden 等价回归）

**Known limitations**: 复核仅能真正读取 .md/.txt/.pdf/.docx 原文（扫描版 PDF 占位）；checkpoint 携带 prompt_version/checkpoint_schema/response_mode 门禁，prompt 或规则变更后旧 checkpoint 拒绝续传。
