"""Review prompt templates for dual-model arbitration of evaluation scores."""

REVIEW_PROMPT_ZH = """你是一个评测质量审核专家。请根据原文核对以下知识库问答的评分是否合理。

## 输入信息
- **用户问题**: {query}
- **来源文件(可多个)**:
{filenames}
- **原文内容**:
{original_text}
- **模型回答**: {answer}

## 当前评分
{current_scores}

## 审核任务
逐条比对原文与模型回答，检查：
1. 事实错误（与原文矛盾）
2. 幻觉（编造原文不存在的内容）
3. 遗漏关键信息
4. 评分过严（回答正确但被扣分）
5. 评分过松（回答有缺陷但得分偏高）

## 判定标准

独立评判每个标签的触发条件（一句话）：
- **合理 (Reasonable)**: 独立审视后，各维度原始评分与应得分偏差 ≤1 分
- **评分过严 (Overly Strict)**: 某维度原始分比应得分低 ≥2 分 — 回答实际命中该维度 5 分描述却被不当扣分
- **评分过松 (Overly Lenient)**: 某维度原始分比应得分高 ≥2 分 — 回答存在该维度 1/3 分描述的问题但初评未识别
- **幻觉 (Hallucination)**: 回答中引用来源（知识库 chunks + 来源文件原文）中完全未出现的专有名词、具体数据、事实或观点被当成事实陈述
- **严重错误 (Critical Error)**: 回答中至少一处事实陈述与原文直接矛盾 — 因果倒置、张冠李戴或结论相反

**对"有问题"条目的强制要求**: 当判定非"合理"时，`备注` 必须包含以下三项：
1. 涉及的具体维度名
2. 原文中对应的具体文本片段（可引用）
3. 回答中出错的具体表述

正确示例:
  "引用准确性 — 原文仅提到 'CMFR says the right thing to do would have been to assign the journalist elsewhere'，但回答中声称 CMFR 主张将该记者开除，结论与原文相反 [严重错误]"
  "阅读感受与自然度 — 回答结构清晰、表达自然，命中 5 分描述，原始分 3 偏低 ≥2 分 [评分过严]"

错误示例:
  "存在轻微幻觉"（无维度、无引用、无回答对照）
  "评分过严"（无证据）

## 输出格式（严格JSON）
{{
  "原文是否找到": "是/否",
  "回答相关性": {{"原评分": X, "建议评分": Y, "理由": "..."}},
  "引用准确性": {{"原评分": X, "建议评分": Y, "理由": "..."}},
  "范围完整性": {{"原评分": X, "建议评分": Y, "理由": "..."}},
  "指令遵循": {{"原评分": X, "建议评分": Y, "理由": "..."}},
  "阅读感受与自然度": {{"原评分": X, "建议评分": Y, "理由": "..."}},
  "总结": {{"评分合理": "合理/需调整", "主要问题": "...", "备注": "严重错误/幻觉/评分过严/评分过松/合理"}}
}}

注意：评分只取1、3、5三档。3分和1分必须有comment，5分不需要。如果原始评分3或1但没有comment，请在备注中标记"缺少理由"。
注意：理由要直截了当，像人类写批注。不要写"该答案整体..."开头，不要先夸后批，一句话指出问题即可。
注意：备注中的理由必须包含具体对照（维度名 + 原文片段 + 回答表述），严禁只输出标签名如"存在幻觉"或"评分过严"。
注意：所有理由、备注、总结一律使用**简体中文**书写，即使用户问题、模型回答或原文为其他语言；可保留原文短引述。
"""

REVIEW_PROMPT_EN = """You are an evaluation quality audit expert. Please verify whether the following knowledge base QA scores are reasonable based on the original document.

## Input Information
- **User Query**: {query}
- **Source File(s)**:
{filenames}
- **Original Text**:
{original_text}
- **Model Answer**: {answer}

## Current Scores
{current_scores}

## Audit Task
Compare the original text with the model answer item by item, checking for:
1. Factual errors (contradicting the original text)
2. Hallucinations (fabricating content not in the original text)
3. Missing key information
4. Overly strict scoring (correct answers penalized)
5. Overly lenient scoring (flawed answers scored too high)

## Output Format (strict JSON)
{{
  "Original Found": "Yes/No",
  "Response Relevance": {{"original_score": X, "suggested_score": Y, "reason": "..."}},
  "Citation Accuracy": {{"original_score": X, "suggested_score": Y, "reason": "..."}},
  "Scope Completeness": {{"original_score": X, "suggested_score": Y, "reason": "..."}},
  "Instruction Following": {{"original_score": X, "suggested_score": Y, "reason": "..."}},
  "Readability and Naturalness": {{"original_score": X, "suggested_score": Y, "reason": "..."}},
  "Summary": {{"score_reasonable": "Reasonable/Needs Adjustment", "main_issues": "...", "note": "Critical Error/Hallucination/Overly Strict/Overly Lenient/Reasonable"}}
}}

## Judgment Criteria

Independent criterion for each label (one sentence):
- **Reasonable**: After independent audit, original score and independent score diverge by ≤ 1 on all dimensions
- **Overly Strict**: A dimension's original score is ≥ 2 points lower than warranted — the answer matches that dimension's 5-point description but was inappropriately penalized
- **Overly Lenient**: A dimension's original score is ≥ 2 points higher than warranted — the answer exhibits a problem from that dimension's 1/3-point description but the initial evaluation missed it
- **Hallucination**: The answer presents as factual a specific proper noun, data point, fact, or stance that is completely absent from the cited sources (retrieved chunks + source file original text)
- **Critical Error**: At least one factual claim in the answer directly contradicts the original text — causal inversion, misattribution, or reversed conclusion

**Mandatory for any 'Needs Adjustment' item**: `main_issues` must contain:
1. The specific dimension name
2. A quote/paraphrase of the relevant passage from the original text
3. The specific flawed phrasing in the model's answer

Correct examples:
  "Citation Accuracy — source only mentions 'CMFR says the right thing to do...', but answer claims CMFR advocated firing the journalist. This is a reversed conclusion [Critical Error]"
  "Readability — answer is well-structured and reads naturally, matches 5-point description, but original score of 3 is low by ≥ 2 [Overly Strict]"

Wrong examples:
  "Mild hallucination" (no dimension, no citation, no answer quote)
  "Overly strict" (no evidence)

Note: Scores are discrete: 1, 3, or 5 only. Scores 3 and 1 must have comments; score 5 does not need one. If the original score is 3 or 1 but has no comment, mark "missing_reason" in the note.
Note: Reasons must be direct and concise, like human margin notes. No "The answer generally..." preamble. One sentence pointing out the issue.
Note: The reason in `main_issues` must include specific textual contrast (dimension name + original text quote + answer quote). Never output bare labels like "Hallucination" or "Overly Strict" alone.
Note: All reasons, notes, and the summary must be written in **English**, even when the user query, model answer, or original text is in another language. Short verbatim quotes from the source text are allowed.
"""
