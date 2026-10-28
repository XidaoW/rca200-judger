"""rcaeval.scoring.process — Process (P) score: 3-dimension LLM judge.

Ported from cms2.0-route-agent/vibeops/eval/rca_evaluate/engine/scoring/
rca_process_scorer.py.

Key brise rule preserved verbatim (`_score_review_dimension`, lines 506-522):
    issues 中 high 严重度计数 → 维度分:
        ≥3 high  → 0
         2 high  → 20
         1 high  → 50
         0 high, low > 0 → 70
         无 issue        → 100
    总分 = 三维均值 / 100

Stripped from brise:
  - vibeops.agent.Agent / vibeops.eval.trace.runtime.model_factory — replaced
    with rcaeval.llm.LLMClient interface
  - vibeops.eval.benchmark_pool.cost_tracker — no-op
  - vibeops.observability.trace_decorator — no-op
  - vibeops.eval.rca_evaluate.engine.fault_signal_library_loader — caller
    passes `causal_chain_text` + `checkpoints_text` directly (M1 keeps it
    simple; M2/M3 wire fault_signal library if needed)
  - "compatible" LLM response normalization (title/content/evidence alias
    handling) — M1 accepts the canonical schema only; LLMs that drift get a
    parse error and the caller retries
"""

from __future__ import annotations

import json
import re
import time
from importlib.resources import files
from typing import Any

from ..llm.base import LLMClient, LLMMessage
from .models import ProcessScoreEvidence

# Brise default cutoff (rca_process_scorer.py:21)
_MAX_ANALYSIS_CHARS = 12_000


# ---------------------------------------------------------------------------
# Score table — verbatim from brise rca_process_scorer.py:506-522
# ---------------------------------------------------------------------------
def _score_review_dimension(issues: list) -> int:
    """单维度评分：无缺陷→100，仅低→70，1高→50，2高→20，≥3高→0.

    Tolerates 3 shapes for each item: (1) brise-canonical dict with explicit
    `severity` key, (2) plain string description (treated as severity=low —
    agent reported an issue but didn't grade it), (3) dict without severity
    (treated as severity=low for the same reason). Anything not dict|str is
    skipped.
    """

    def sev(i) -> str:
        if isinstance(i, dict):
            s = (i.get("severity") or "").lower()
            return s if s in ("high", "low") else "low"
        if isinstance(i, str) and i.strip():
            return "low"
        return ""

    sevs = [sev(i) for i in issues]
    high = sum(1 for s in sevs if s == "high")
    low = sum(1 for s in sevs if s == "low")
    if high >= 3:
        return 0
    if high == 2:
        return 20
    if high == 1:
        return 50
    if low > 0:
        return 70
    return 100


def _first_text(*values: object) -> str:
    return next(
        (str(value).strip() for value in values if str(value or "").strip()), ""
    )


def _normalize_dimensions(raw: object) -> list[dict[str, Any]]:
    """Normalize current Brise list/dict shapes and known field aliases."""
    if isinstance(raw, dict):
        items = [
            {"dimension": name, **value}
            for name, value in raw.items()
            if isinstance(value, dict)
        ]
    elif isinstance(raw, list):
        items = [item for item in raw if isinstance(item, dict)]
    else:
        items = []
    normalized: list[dict[str, Any]] = []
    for item in items:
        strengths = []
        for value in item.get("strengths") or []:
            if isinstance(value, dict):
                text = _first_text(
                    value.get("content"), value.get("comment"), value.get("evidence")
                )
            else:
                text = str(value).strip()
            if text:
                strengths.append(text)
        issues = []
        for value in item.get("issues") or []:
            if isinstance(value, str):
                issues.append({"severity": "low", "description": value})
                continue
            if not isinstance(value, dict):
                continue
            issues.append(
                {
                    "severity": value.get("severity") or "low",
                    "description": _first_text(
                        value.get("description"),
                        value.get("title"),
                        value.get("content"),
                        value.get("explanation"),
                        value.get("evidence"),
                    ),
                    "issue_type": _first_text(
                        value.get("issue_type"), value.get("title")
                    ),
                    "missing": _first_text(
                        value.get("missing"),
                        value.get("explanation"),
                        value.get("content"),
                        value.get("evidence"),
                    ),
                }
            )
        normalized.append(
            {
                "dimension": item.get("dimension") or item.get("name") or "",
                "strengths": strengths,
                "issues": issues,
            }
        )
    return normalized


def _build_review_evidence(
    parsed: dict,
    expected_fault_type: str,
) -> ProcessScoreEvidence:
    """Build ProcessScoreEvidence from a compatible review payload."""
    normalized = _normalize_dimensions(parsed.get("dimensions"))
    dim_map: dict[str, dict] = {item.get("dimension", ""): item for item in normalized}
    required = {"causal_contradiction", "logic_jump", "evidence_gap"}
    if len(normalized) != 3 or set(dim_map) != required:
        raise ValueError(
            "process judge must return exactly three unique dimensions: "
            "causal_contradiction, logic_jump, evidence_gap"
        )

    dim_scores: list[int] = []
    review_dimensions: list[dict] = []
    for dim_name in ("causal_contradiction", "logic_jump", "evidence_gap"):
        dim = dim_map.get(dim_name) or {}
        issues = dim.get("issues") or []
        strengths = dim.get("strengths") or []
        dim_score = _score_review_dimension(issues)
        dim_scores.append(dim_score)
        review_dimensions.append(
            {
                "dimension": dim_name,
                "score": dim_score,
                "strengths": list(strengths),
                "issues": [
                    {
                        "severity": (
                            i.get("severity", "low") if isinstance(i, dict) else "low"
                        ),
                        "description": (
                            i.get("description", "") if isinstance(i, dict) else str(i)
                        ),
                        "issue_type": (
                            i.get("issue_type", "") if isinstance(i, dict) else ""
                        ),
                        "missing": (
                            i.get("missing", "") if isinstance(i, dict) else ""
                        ),
                    }
                    for i in issues[:10]
                ],
            }
        )

    avg = sum(dim_scores) / len(dim_scores) if dim_scores else 0
    score = round(avg / 100, 4)
    reasoning = parsed.get("reasoning", "") or ""

    return ProcessScoreEvidence(
        score=score,
        score_source="rca_review",
        tier="",
        tier_reason=reasoning,
        review_dimensions=review_dimensions,
        explanation_for_human=f"过程 {round(avg)}/100（三维审查）",
        fault_type_used=expected_fault_type,
    )


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------
def _load_prompt() -> str:
    """Load RCA_PROCESS_REVIEW_PROMPT shipped under rcaeval/data/prompts/."""
    return (
        files("brise_rca_score.data.prompts") / "RCA_PROCESS_REVIEW_PROMPT.md"
    ).read_text(encoding="utf-8")


def _strip_markdown_json_fence(text: str) -> str:
    """Some LLMs wrap JSON in ```json ... ```. Strip the fence."""
    m = re.match(r"\s*```(?:json|jsonc)?\s*\n(.*?)\n```\s*$", text, re.S)
    return m.group(1) if m else text


def _iter_json_objects(text: str):
    """Yield independently decodable JSON objects from a model response.

    Reasoning models may prepend prose or a worked JSON example before their
    final answer.  Decoding the whole response (or slicing it greedily from
    the first opening brace to the last closing brace) rejects such otherwise
    valid responses.  The caller still performs the strict process schema
    validation, so this only improves framing compatibility.
    """
    decoder = json.JSONDecoder()
    cursor = 0
    while True:
        start = text.find("{", cursor)
        if start < 0:
            return
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        cursor = end
        if isinstance(value, dict):
            yield value


# ---------------------------------------------------------------------------
# Top-level scorer
# ---------------------------------------------------------------------------
def score_process(
    *,
    llm: LLMClient,
    analysis_output: str,
    expected_fault_type: str,
    fault_desc: str = "",
    injection_target: str = "",
    causal_chain_text: str = "",
    checkpoints_text: str = "",
    topology_summary: str = "",
    entity_attribution_hint: str = "",
    tool_calls_summary: str = "",
    agent_final_conclusion: str = "",
    evidence_text: str = "",
    model: str = "",
    temperature: float = 0.0,
    retry_instruction: str = "",
    structured_retry: bool = False,
) -> ProcessScoreEvidence:
    """RCA process scoring entry point (mirrors brise score_process_complete).

    Args:
        llm: any LLMClient impl (StubLLM for tests, DashScopeLLM in prod).
        analysis_output: Agent's raw RCA analysis text (will be truncated to
            12k chars to match brise).
        expected_fault_type: GT fault_id (e.g. "F010-slowSQL").
        fault_desc: human description of the GT fault (optional).
        injection_target: GT root-cause entity name (e.g. "inventory").
        causal_chain_text / checkpoints_text: GT reasoning-step summaries
            (brise loads these from fault_signal_library; rcaeval M1 expects
            caller to pre-render and pass in).
        topology_summary, entity_attribution_hint: optional context, appended
            to the prompt suffix as brise does (lines 341-349 of source).

    Returns:
        ProcessScoreEvidence with score ∈ [0, 1] and full review_dimensions[].

    Raises:
        ValueError if the LLM response can't be parsed as the expected JSON
        schema; caller should retry or fall back to a stub.
    """
    if not analysis_output or not analysis_output.strip():
        return ProcessScoreEvidence(
            score=0.0,
            score_source="skipped",
            tier_reason="no analysis_output",
            review_dimensions=[],
            explanation_for_human="跳过（无分析文本）",
            fault_type_used=expected_fault_type,
        )

    prompt = _load_prompt().format(
        injected_fault=expected_fault_type or "unknown",
        injected_fault_desc=fault_desc,
        injected_entity=injection_target or "未知",
        expected_causal_chain=causal_chain_text,
        expected_checkpoints=checkpoints_text,
        agent_output=analysis_output[:_MAX_ANALYSIS_CHARS],
    )
    if agent_final_conclusion:
        prompt += f"\n\n## Agent 最终结论（定因侧结构化抽取）\n{agent_final_conclusion[:2000]}"
    if evidence_text:
        prompt += f"\n\n## 当前根因结论的高亮证据域\n{evidence_text[:4000]}"
    if tool_calls_summary:
        safe_summary = tool_calls_summary.replace("`", "'")
        prompt += (
            "\n\n## Agent 工具调用摘要（从 trajectory 原始工具返回程序判定）\n"
            "```tool_calls_data\n" + safe_summary + "\n```\n"
            "失败或空结果不能作为正常、无异常、已排除或已核查的依据。"
        )
    if topology_summary:
        prompt += f"\n\n## Agent 分析涉及的服务拓扑\n{topology_summary}"
    if entity_attribution_hint:
        prompt += (
            f"\n\n## 定界评分参考（已完成）\n{entity_attribution_hint}\n"
            f"注意：如果定界评分已判定 Agent 归因路径合理，"
            f"因果矛盾维度不应对此部分判定为'方向错误'。"
        )
    if retry_instruction:
        prompt += f"\n\n## 输出协议纠正\n{retry_instruction}"

    messages = [
        LLMMessage(
            role="system",
            content=(
                "你是 RCA 过程三维审查评估器。"
                "请严格依据输入中的分析文本、预期因果链和排查节点，"
                "输出结构化 JSON，不要编造未出现的证据。"
            ),
        ),
        LLMMessage(role="user", content=prompt),
    ]
    start_ms = int(time.time() * 1000)
    chat = (
        getattr(llm, "chat_structured_retry", llm.chat)
        if structured_retry
        else llm.chat
    )
    raw = chat(messages, model=model, temperature=temperature, response_format="json")
    _ = int(time.time() * 1000) - start_ms

    body = _strip_markdown_json_fence(raw)
    saw_object = False
    last_schema_error: ValueError | None = None
    top_keys: list[str] = []
    for parsed in _iter_json_objects(body):
        saw_object = True
        top_keys = sorted(str(key) for key in parsed)
        if "dimensions" not in parsed:
            continue
        try:
            return _build_review_evidence(parsed, expected_fault_type)
        except ValueError as exc:
            # A reasoning preamble may contain a schema/example object before
            # the final answer.  Keep scanning, but preserve the last strict
            # schema error if no valid answer follows.
            last_schema_error = exc
    if last_schema_error is not None:
        raise last_schema_error
    if saw_object:
        raise ValueError(
            "LLM response missing 'dimensions': "
            f"top_type=dict; top_keys={top_keys}; response_chars={len(body)}"
        )
    raise ValueError(
        "LLM response not valid JSON: no decodable object; "
        f"response_chars={len(body)}"
    )


# ---------------------------------------------------------------------------
# Offline path: caller already has review_dimensions[] (e.g. from brise output),
# skip the LLM and score directly. Useful for replaying historical traces.
# ---------------------------------------------------------------------------
def score_process_from_review_dimensions(
    review_dimensions: list[dict[str, Any]],
    *,
    expected_fault_type: str = "",
    reasoning: str = "",
) -> ProcessScoreEvidence:
    """Build P score from a pre-computed dimensions list (no LLM call).

    Each item must have: dimension, strengths (optional), issues (list of
    {severity, ...}). brise's evaluate-get output's `details.process.review_dimensions`
    is directly usable here.
    """
    return _build_review_evidence(
        {"dimensions": review_dimensions, "reasoning": reasoning},
        expected_fault_type,
    )
