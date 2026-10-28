"""rcaeval.scoring.fault_type — F (fault_type) scorer.

Ported from cms2.0-route-agent/vibeops/eval/rca_evaluate/engine/scoring/
fault_type_scorer.py + causal_cluster.py + llm_causal_judge.py.

Pipeline (matches brise verbatim, lines 100-266 of fault_type_scorer.py):

    1. Normalize: strip + lowercase + (optional) dictionary alias resolve
    2. EXACT  : expected_norm == any actual_norm → tier=EXACT (1.0)
    3. CLUSTER: both in same fault_causal_clusters → tier=CLUSTER (0.5),
                shared_category=cluster_label
    4. GT causal chain: walk gt_causal_chain ReasoningStep[], hop k →
                tier=CAUSAL_KHOP (k clamped to 1/2/3), score from tier table
                (1HOP=0.85, 2HOP=0.65, 3HOP=0.50)
    5. LLM rescue: if 1-4 all miss, call LLM judge with the GT chain context;
                map relation_type → score:
                  causal_downstream_1hop → 0.50
                  causal_downstream_2hop → 0.35
                  causal_upstream        → 0.50
                  unrelated              → 0.00
                confidence < 0.4 → reject (treat as None)
    6. Enumeration penalty: actual_count > 1 AND tier != EXACT → score × (1/√n)

Stripped:
  - vibeops.eval.rca_evaluate.engine.fault_catalog_loader.lookup_fault_type_description
    → caller passes fault_descriptions dict (optional)
  - vibeops.eval.benchmark_pool.cost_tracker → no-op
  - vibeops.observability.trace_decorator → no-op
  - structured_agent_chain serialization (only used by LLM rescue prompt
    context; M2 lets caller pass agent_chain_text as a string directly)
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from importlib.resources import files

from ..llm.base import LLMClient, LLMMessage
from .models import (
    _FAULT_TYPE_TIER_SCORES,
    FaultTypeScoreEvidence,
    FaultTypeTier,
)


# ---------------------------------------------------------------------------
# Causal cluster config — verbatim port of brise causal_cluster.py
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClusterMatchResult:
    cluster_name: str
    cluster_label: str
    matched_actual: str


@dataclass(frozen=True)
class CausalClusterConfig:
    clusters: dict[str, list[str]] = field(default_factory=dict)
    cluster_labels: dict[str, str] = field(default_factory=dict)
    fault_id_to_cluster: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_dict(raw: dict) -> "CausalClusterConfig":
        clusters: dict[str, list[str]] = {}
        labels: dict[str, str] = {}
        fid_to_cluster: dict[str, str] = {}
        for name, info in raw.get("clusters", {}).items():
            ids = [fid.strip().lower() for fid in info.get("fault_ids", [])]
            clusters[name] = ids
            labels[name] = info.get("label", name)
            for fid in ids:
                fid_to_cluster[fid] = name
        return CausalClusterConfig(
            clusters=clusters,
            cluster_labels=labels,
            fault_id_to_cluster=fid_to_cluster,
        )


_CLUSTER_CACHE: CausalClusterConfig | None = None


def load_causal_clusters(path: str | None = None) -> CausalClusterConfig:
    """Load fault_causal_clusters.json. Cached on first call."""
    global _CLUSTER_CACHE
    if _CLUSTER_CACHE is not None and path is None:
        return _CLUSTER_CACHE
    if path is None:
        raw = json.loads(
            (files("brise_rca_score.data") / "fault_causal_clusters.json").read_text(
                encoding="utf-8"
            )
        )
    else:
        from pathlib import Path

        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = CausalClusterConfig.from_dict(raw)
    if path is None:
        _CLUSTER_CACHE = cfg
    return cfg


def match_causal_cluster(
    *,
    expected: str,
    actuals: list[str],
    config: CausalClusterConfig,
) -> ClusterMatchResult | None:
    if not config.fault_id_to_cluster:
        return None
    exp_key = expected.strip().lower()
    exp_cluster = config.fault_id_to_cluster.get(exp_key)
    if not exp_cluster:
        return None
    for actual in actuals:
        act_key = actual.strip().lower()
        act_cluster = config.fault_id_to_cluster.get(act_key)
        if act_cluster == exp_cluster:
            return ClusterMatchResult(
                cluster_name=exp_cluster,
                cluster_label=config.cluster_labels.get(exp_cluster, exp_cluster),
                matched_actual=actual,
            )
    return None


# ---------------------------------------------------------------------------
# Normalize: dictionary alias + Fxxx-canonical extraction
# ---------------------------------------------------------------------------
def _to_fid(raw: str, dictionary: dict[str, str] | None) -> str:
    """Resolve `raw` to canonical fault_id.

    Brise loads a full FaultTypeDictionaryConfig. This standalone scorer accepts
    an optional caller-provided ``dict[alias_lowercase, canonical_fault_id]``.
    """
    if not raw:
        return raw
    key = raw.strip().lower()
    if dictionary and key in dictionary:
        return dictionary[key]
    return raw


# ---------------------------------------------------------------------------
# GT causal-chain walk (brise: _match_against_gt_causal_chain, 269-364)
# ---------------------------------------------------------------------------
@dataclass
class ReasoningStep:
    """Minimal port of brise gt_models.ReasoningStep.

    Brise GT carries a list of these per case (visible in evaluate API response
    as ground_truth.reasoning_steps). We only need the fields scoring touches.
    """

    step: int = 0
    target: str = ""
    fault_id: str = ""
    step_type: str = ""  # cause | propagation | impact (brise: StepType enum)
    title: str = ""
    description: str = ""
    observability: list[dict] = field(default_factory=list)


def _match_gt_chain(
    *,
    gt_chain: list[ReasoningStep],
    actual_raw: list[str],
    actual_norm: list[str],
    dictionary: dict[str, str] | None,
) -> tuple[str, FaultTypeTier, str, int] | None:
    """Walk GT reasoning_steps, find best (lowest hop) match.

    Returns: (matched_raw_actual, tier, chain_display, depth) or None.
    Hop k = max(0, step - 1) → step=1 (root) → EXACT, step=2 → 1HOP, ...
    """
    if not gt_chain:
        return None
    chain_display_parts = [
        f"{s.target or '?'}: {s.fault_id or '?'}"
        for s in sorted(gt_chain, key=lambda x: x.step)
    ]
    chain_display = " → ".join(chain_display_parts)

    best_depth = 10_000
    best_actual_raw = ""
    best_tier = FaultTypeTier.NONE

    for idx, (raw_a, norm_a) in enumerate(zip(actual_raw, actual_norm)):
        act_keys: set[str] = {norm_a.strip().lower()}
        # F010-slowSQL → also try "slowsql" as a key
        for k in [norm_a, raw_a]:
            kl = k.strip().lower()
            if "-" in kl:
                act_keys.add(kl.split("-", 1)[1])
            act_keys.add(kl)

        for step in gt_chain:
            step_keys: set[str] = set()
            if step.fault_id:
                step_keys.add(step.fault_id.strip().lower())
                if "-" in step.fault_id:
                    step_keys.add(step.fault_id.split("-", 1)[1].strip().lower())
            if act_keys & step_keys:
                depth = max(0, step.step - 1)
                if depth < best_depth:
                    best_depth = depth
                    best_actual_raw = raw_a
                    if depth == 0:
                        best_tier = FaultTypeTier.EXACT
                    else:
                        best_tier = {
                            1: FaultTypeTier.CAUSAL_1HOP,
                            2: FaultTypeTier.CAUSAL_2HOP,
                        }.get(min(depth, 3), FaultTypeTier.CAUSAL_3HOP)
                break

    if best_actual_raw:
        return (best_actual_raw, best_tier, chain_display, best_depth)
    return None


# ---------------------------------------------------------------------------
# LLM rescue — verbatim port of judge_fault_type_causal_by_llm + scoring map
# brise: llm_causal_judge.py
# ---------------------------------------------------------------------------
@dataclass
class LlmCausalJudgeResult:
    has_causal_relation: bool
    relation_type: str  # causal_downstream_1hop / _2hop / causal_upstream / unrelated
    confidence: float
    best_actual: str = ""
    explanation: str = ""
    score: float = 0.0
    virtual_chain_path: list[dict] = field(default_factory=list)
    chain_direction: str = ""  # downstream | upstream


_LLM_RESCUE_SCORE_MAP: dict[str, float] = {
    "causal_downstream_1hop": 0.50,
    "causal_downstream_2hop": 0.35,
    "causal_upstream": 0.50,
    "unrelated": 0.00,
}
_LLM_RESCUE_CONFIDENCE_THRESHOLD = 0.4
_LLM_RESCUE_MAX_ACTUALS = 5
_DEPTH_LABELS = {0: "根因", 1: "机制", 2: "传播"}


def _format_causal_chain_for_llm(gt_chain: list[ReasoningStep] | None) -> str:
    if not gt_chain:
        return "（无因果链数据）"
    blocks = []
    for step in sorted(gt_chain, key=lambda x: x.step):
        depth = max(0, step.step - 1)
        depth_label = _DEPTH_LABELS.get(depth, "症状")
        header = f"Step {step.step} [{depth_label}] {step.title or f'step_{step.step}'}"
        meta = []
        if step.fault_id:
            meta.append(f"fault_id={step.fault_id}")
        if step.target:
            meta.append(f"target={step.target}")
        if step.step_type:
            meta.append(f"step_type={step.step_type}")
        lines = [header]
        if meta:
            lines.append("  " + "  ".join(meta))
        if step.description:
            lines.append(f"  描述：{step.description.strip()}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _load_rescue_prompt() -> str:
    return (
        files("brise_rca_score.data.prompts") / "RCA_LLM_CAUSAL_JUDGE_PROMPT.md"
    ).read_text(encoding="utf-8")


def _parse_llm_rescue_response(
    raw_text: str,
    actuals: list[str],
) -> LlmCausalJudgeResult | None:
    """Parse {steps, best_actual, relation, confidence, chain, chain_path, chain_direction}."""
    # Reasoning models may emit prose (or a worked JSON example) before the
    # final answer.  A greedy ``\{.*\}`` slice joins the first brace in that
    # prose to the last brace in the final answer and turns two valid fragments
    # into one invalid document.  Scan for independently decodable JSON values
    # and accept only an object that passes the existing relation whitelist.
    text = raw_text.strip()
    decoder = json.JSONDecoder()
    cursor = 0
    while True:
        start = text.find("{", cursor)
        if start < 0:
            return None
        try:
            body, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        cursor = end
        if not isinstance(body, dict):
            continue

        relation = str(body.get("relation", "")).strip()
        if relation not in _LLM_RESCUE_SCORE_MAP:
            continue
        try:
            confidence = float(body.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if confidence < _LLM_RESCUE_CONFIDENCE_THRESHOLD and relation != "unrelated":
            # Trust unrelated even with low confidence (negative case); reject positive
            return LlmCausalJudgeResult(
                has_causal_relation=False,
                relation_type=relation,
                confidence=confidence,
                explanation=str(body.get("chain", "") or ""),
            )

        score = _LLM_RESCUE_SCORE_MAP[relation]
        best = str(body.get("best_actual", "") or "").strip()
        if not best and actuals:
            best = actuals[0]
        chain_path = body.get("chain_path", []) or []
        if not isinstance(chain_path, list):
            chain_path = []

        return LlmCausalJudgeResult(
            has_causal_relation=(relation != "unrelated"),
            relation_type=relation,
            confidence=confidence,
            best_actual=best,
            explanation=str(body.get("chain", "") or ""),
            score=score,
            virtual_chain_path=chain_path,
            chain_direction=str(body.get("chain_direction", "") or ""),
        )


def judge_fault_type_causal_by_llm(
    *,
    llm: LLMClient,
    expected: str,
    actuals: list[str],
    expected_desc: str = "",
    actual_descs: dict[str, str] | None = None,
    causal_steps: list[ReasoningStep] | None = None,
    agent_chain_text: str = "",
    agent_conclusion: str = "",
    model: str = "",
    temperature: float = 0.0,
) -> LlmCausalJudgeResult | None:
    """Run causal rescue; evaluator transport/protocol failures are explicit."""
    if not expected or not actuals:
        return None
    truncated = actuals[:_LLM_RESCUE_MAX_ACTUALS]

    actual_lines = []
    for a in truncated:
        desc = (actual_descs or {}).get(a.strip().lower(), "")
        actual_lines.append(f"  - {a}" + (f"（{desc}）" if desc else ""))

    causal_chain_text = _format_causal_chain_for_llm(causal_steps)
    prompt = _load_rescue_prompt().format(
        expected=expected,
        expected_desc=expected_desc or "未知",
        causal_chain_text=causal_chain_text,
        actual_list="\n".join(actual_lines),
        agent_chain_text=agent_chain_text or "（无 Agent 传播链）",
        agent_conclusion=agent_conclusion or "（无结论）",
    )

    for attempt in range(3):
        retry_instruction = ""
        if attempt:
            retry_instruction = (
                "\n\n上一次响应未满足结构化协议。只输出一个 JSON 对象；"
                "relation 必须使用提示词允许的枚举值，confidence 必须是数字。"
                "不要输出示例、Markdown 或额外说明。"
            )
        messages = [LLMMessage(role="user", content=prompt + retry_instruction)]
        chat = (
            getattr(llm, "chat_structured_retry", llm.chat) if attempt > 0 else llm.chat
        )
        raw = chat(
            messages, model=model, temperature=temperature, response_format="json"
        )
        result = _parse_llm_rescue_response(raw, truncated)
        if result is not None:
            return result
    raise ValueError("fault causal judge returned an invalid response after 3 attempts")


# ---------------------------------------------------------------------------
# Top-level F scorer
# ---------------------------------------------------------------------------
def score_fault_type(
    *,
    expected_fault_type: str,
    actual_fault_types: list[str],
    actual_confidences: list[float] | None = None,
    dictionary: dict[str, str] | None = None,
    gt_causal_chain: list[ReasoningStep] | None = None,
    cluster_config: CausalClusterConfig | None = None,
    # LLM rescue inputs (optional)
    llm: LLMClient | None = None,
    expected_desc: str = "",
    actual_descs: dict[str, str] | None = None,
    agent_chain_text: str = "",
    agent_conclusion: str = "",
) -> FaultTypeScoreEvidence:
    """Score fault_type with full brise pipeline (EXACT → CLUSTER → GT chain → LLM rescue)."""
    expected_raw = (expected_fault_type or "").strip()
    actual_raw = [a.strip() for a in (actual_fault_types or []) if a.strip()]
    expected_norm = _to_fid(expected_raw, dictionary) if expected_raw else expected_raw
    actual_norm = [_to_fid(a, dictionary) for a in actual_raw]

    # Empty fast-path
    if not expected_raw or not actual_raw:
        return FaultTypeScoreEvidence(
            expected_raw=expected_raw,
            expected_normalized=expected_norm,
            actual_raw=actual_raw,
            match_tier=FaultTypeTier.NONE.value,
            match_detail="缺少预期或实际故障类型，无法评分。",
            score=0.0,
            scoring_method="unmatched",
            predicted_fault_ids=actual_norm,
        )

    # ---- EXACT ----
    exp_key = expected_norm.strip().lower()
    for idx, norm_a in enumerate(actual_norm):
        if exp_key == norm_a.strip().lower():
            raw_a = actual_raw[idx] if idx < len(actual_raw) else norm_a
            return FaultTypeScoreEvidence(
                expected_raw=expected_raw,
                expected_normalized=expected_norm,
                actual_raw=actual_raw,
                predicted_fault_ids=actual_norm,
                matched_fault_id=raw_a,
                match_tier=FaultTypeTier.EXACT.value,
                match_detail=f"精确匹配：预期 {expected_raw}，实际 {raw_a}。",
                score=round(_FAULT_TYPE_TIER_SCORES[FaultTypeTier.EXACT], 4),
                scoring_method="exact",
            )

    # ---- CLUSTER ----
    cfg = cluster_config or load_causal_clusters()
    cluster_result = match_causal_cluster(
        expected=exp_key,
        actuals=[n.strip().lower() for n in actual_norm],
        config=cfg,
    )
    if cluster_result is not None:
        matched_raw = cluster_result.matched_actual
        for i, n in enumerate(actual_norm):
            if n.strip().lower() == cluster_result.matched_actual:
                matched_raw = actual_raw[i] if i < len(actual_raw) else n
                break
        evidence = FaultTypeScoreEvidence(
            expected_raw=expected_raw,
            expected_normalized=expected_norm,
            actual_raw=actual_raw,
            predicted_fault_ids=actual_norm,
            matched_fault_id=matched_raw,
            match_tier=FaultTypeTier.CLUSTER.value,
            match_detail=(
                f"同簇匹配（{cluster_result.cluster_label}）："
                f"预期 {expected_raw}，实际 {matched_raw}。"
            ),
            shared_category=cluster_result.cluster_label,
            score=round(_FAULT_TYPE_TIER_SCORES[FaultTypeTier.CLUSTER], 4),
            scoring_method="cluster",
        )
        # Note: brise does NOT apply enum penalty to EXACT or CLUSTER tiers
        # (see brise fault_type_scorer.py line 191 — return is inside the
        # cluster branch, before _apply_enumeration_penalty at line 264).
        return evidence

    # ---- GT causal chain ----
    chain_hit = _match_gt_chain(
        gt_chain=gt_causal_chain or [],
        actual_raw=actual_raw,
        actual_norm=actual_norm,
        dictionary=dictionary,
    )
    if chain_hit is not None:
        matched_raw, tier, chain_display, depth = chain_hit
        evidence = FaultTypeScoreEvidence(
            expected_raw=expected_raw,
            expected_normalized=expected_norm,
            actual_raw=actual_raw,
            predicted_fault_ids=actual_norm,
            matched_fault_id=matched_raw,
            match_tier=tier.value,
            match_detail=f"GT 因果链匹配（{tier.label}）：{chain_display}",
            score=round(_FAULT_TYPE_TIER_SCORES[tier], 4),
            scoring_method="causal_chain",
            causal_match_depth=depth,
            causal_match_kind="gt_chain",
        )
        _apply_enumeration_penalty(evidence, len(actual_raw))
        return evidence

    # ---- LLM rescue ----
    evidence = FaultTypeScoreEvidence(
        expected_raw=expected_raw,
        expected_normalized=expected_norm,
        actual_raw=actual_raw,
        predicted_fault_ids=actual_norm,
        match_tier=FaultTypeTier.NONE.value,
        match_detail="GT 因果链未命中。",
        score=0.0,
        scoring_method="unmatched",
    )
    if llm is not None:
        llm_result = judge_fault_type_causal_by_llm(
            llm=llm,
            expected=expected_raw,
            actuals=actual_raw,
            expected_desc=expected_desc,
            actual_descs=actual_descs,
            causal_steps=gt_causal_chain,
            agent_chain_text=agent_chain_text,
            agent_conclusion=agent_conclusion,
        )
        if llm_result is not None and llm_result.has_causal_relation:
            # Brise: try the merged-chain refinement first; if it produces a
            # better tier than the flat LLM_RESCUE_SCORE_MAP, use it.
            merged = score_against_merged_chain(
                gt_steps=gt_causal_chain,
                llm_chain_path=llm_result.virtual_chain_path,
                chain_direction=llm_result.chain_direction,
                actual_fault_types=actual_raw,
            )
            if merged is not None:
                evidence.score = round(merged.score, 4)
                evidence.scoring_method = "llm_merged_chain"
                evidence.match_tier = merged.tier
                evidence.matched_fault_id = merged.best_actual
            else:
                evidence.matched_fault_id = llm_result.best_actual
                evidence.score = round(llm_result.score, 4)
                evidence.scoring_method = "llm_causal_rescue"
                evidence.match_tier = f"llm_{llm_result.relation_type}"
            evidence.match_detail = f"LLM 因果补救：{llm_result.explanation[:200]}"
            evidence.causal_match_kind = llm_result.relation_type

    _apply_enumeration_penalty(evidence, len(actual_raw))
    return evidence


def _apply_enumeration_penalty(
    evidence: FaultTypeScoreEvidence, actual_count: int
) -> None:
    """Brise rule: actual_count > 1 AND tier != EXACT → score × 1/√n."""
    if actual_count > 1 and evidence.match_tier != FaultTypeTier.EXACT.value:
        penalty = 1.0 / math.sqrt(actual_count)
        evidence.score = round(evidence.score * penalty, 4)


# ---------------------------------------------------------------------------
# Verbatim ports for full brise parity (non-LLM deterministic logic)
# ---------------------------------------------------------------------------
_MERMAID_RE = re.compile(r"```mermaid\s*(.*?)```", re.DOTALL)
_CONCLUSION_KEYS = ("根因结论", "诊断摘要", "## 结论", "结论：")
_LLM_HOP_SCORE_CAP = 0.50  # LLM virtual-node ceiling, equals causal_3hop floor


def extract_agent_chain_and_conclusion(actual_text: str) -> tuple[str, str]:
    """Extract mermaid chain + conclusion paragraph from agent analysis text.

    Verbatim from brise fault_type_scorer.py:395-423. Returns (chain_text,
    conclusion_text). chain_text is the bare mermaid block content; conclusion
    is the first paragraph after any keyword in _CONCLUSION_KEYS, ≤500 chars,
    falling back to the last 500 chars.
    """
    if not actual_text:
        return "", ""
    chain_text = ""
    m = _MERMAID_RE.search(actual_text)
    if m:
        chain_text = m.group(1).strip()
    conclusion = ""
    text_lower = actual_text.lower()
    for key in _CONCLUSION_KEYS:
        idx = text_lower.find(key.lower())
        if idx >= 0:
            conclusion = actual_text[idx : idx + 500].strip()
            break
    if not conclusion:
        conclusion = actual_text[-500:].strip()
    return chain_text, conclusion


def serialize_structured_agent_chain(nodes: list[dict] | None) -> str:
    """Serialize structured agent chain into the exact brise format.

    Verbatim from brise fault_type_scorer.py:426-440. Each node becomes a line
    `entity_name | detail | fault_id` (skipping empty fields).
    """
    if not nodes:
        return ""
    lines: list[str] = []
    for item in nodes:
        entity = str(item.get("entity_name", "") or "").strip()
        detail = str(item.get("detail", "") or "").strip()
        fault_id = str(item.get("fault_id", "") or "").strip()
        parts = [p for p in [entity, detail, fault_id] if p]
        if parts:
            lines.append(" | ".join(parts))
    return "\n".join(lines)


@dataclass
class _MergedChainMatch:
    score: float
    tier: str
    best_actual: str


def score_against_merged_chain(
    *,
    gt_steps: list[ReasoningStep] | None,
    llm_chain_path: list[dict],
    chain_direction: str,
    actual_fault_types: list[str],
) -> _MergedChainMatch | None:
    """Merge GT chain + LLM virtual chain, score Agent fault_ids by hop distance.

    Verbatim port of brise fault_type_scorer.py:461-525. Merge rules:
      - downstream: GT chain + LLM nodes appended at tail
      - upstream:   LLM nodes prepended at head + GT chain
    Hop score table: {0:1.0, 1:0.85, 2:0.65, 3:0.50, 4:0.40, 5:0.35}, max(0.25,
    0.50 - 0.05 × (hop - 3)) beyond 5. LLM nodes capped at _LLM_HOP_SCORE_CAP.
    """
    if not llm_chain_path or not actual_fault_types:
        return None

    gt_list = sorted(gt_steps or [], key=lambda s: s.step)
    merged_fault_ids: list[str] = []
    is_llm_node: list[bool] = []

    if chain_direction == "upstream":
        for node in llm_chain_path:
            merged_fault_ids.append(str(node.get("fault_id", "") or "").strip().lower())
            is_llm_node.append(True)
        for s in gt_list:
            merged_fault_ids.append((s.fault_id or "").strip().lower())
            is_llm_node.append(False)
    else:
        for s in gt_list:
            merged_fault_ids.append((s.fault_id or "").strip().lower())
            is_llm_node.append(False)
        for node in llm_chain_path:
            merged_fault_ids.append(str(node.get("fault_id", "") or "").strip().lower())
            is_llm_node.append(True)

    _HOP_SCORES = {0: 1.0, 1: 0.85, 2: 0.65, 3: 0.50, 4: 0.40, 5: 0.35}
    best_score = 0.0
    best_actual = ""
    best_tier = ""

    for actual in actual_fault_types:
        actual_key = actual.strip().lower()
        for idx, chain_fid in enumerate(merged_fault_ids):
            if not chain_fid or chain_fid != actual_key:
                continue
            hop = idx
            raw_score = _HOP_SCORES.get(hop, max(0.25, 0.50 - 0.05 * (hop - 3)))
            if is_llm_node[idx]:
                raw_score = min(raw_score, _LLM_HOP_SCORE_CAP)
            if raw_score > best_score:
                best_score = raw_score
                best_actual = actual
                hop_label = (
                    f"merged_{hop}hop" if is_llm_node[idx] else f"causal_{hop}hop"
                )
                best_tier = hop_label

    if best_score > 0:
        return _MergedChainMatch(
            score=best_score, tier=best_tier, best_actual=best_actual
        )
    return None
