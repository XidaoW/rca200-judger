"""Process-judge live scoring and deterministic replay orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from .contracts import Diagnosis, RootCause
from .llm.base import LLMClient
from .resolution import _agent_chain_text, _render_gt_context
from .scoring.fault_type import ReasoningStep
from .scoring.models import (
    EntityScoreEvidence,
    FaultTypeScoreEvidence,
    ProcessScoreEvidence,
)
from .scoring.process import score_process, score_process_from_review_dimensions


def _process_formula(evidence: ProcessScoreEvidence) -> str:
    scores = [int(item.get("score") or 0) for item in evidence.review_dimensions]
    if not scores:
        return ""
    joined = " + ".join(str(score) for score in scores)
    return f"({joined})/{len(scores)} = {round(evidence.score * 100)}"


def _call_process_judge(
    *,
    llm: LLMClient,
    diagnosis: Diagnosis,
    root: RootCause,
    chain: str,
    checkpoints: str,
    model: str,
    temperature: float,
) -> ProcessScoreEvidence:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return score_process(
                llm=llm,
                analysis_output=diagnosis.analysis_output or diagnosis.conclusion.text,
                expected_fault_type=root.fault_id,
                fault_desc=root.conclusion,
                injection_target=(
                    root.entity_group.entity_refs[0].entity_name
                    if root.entity_group.entity_refs
                    else ""
                ),
                causal_chain_text=chain,
                checkpoints_text=checkpoints,
                entity_attribution_hint=_agent_chain_text(diagnosis),
                tool_calls_summary=diagnosis.tool_calls_summary,
                agent_final_conclusion=diagnosis.conclusion.text,
                evidence_text=diagnosis.conclusion.text,
                model=model,
                temperature=temperature,
                retry_instruction=(
                    ""
                    if attempt == 0
                    else (
                        "上一次响应无法通过严格 schema 校验。只输出一个 JSON 对象，"
                        "且 dimensions 必须且只能包含 causal_contradiction、"
                        "logic_jump、evidence_gap 三个唯一维度；"
                        "不要输出示例、Markdown 或额外说明。"
                    )
                ),
                structured_retry=attempt > 0,
            )
        except (RuntimeError, ValueError) as exc:
            last_error = exc
    detail = "unknown"
    if last_error is not None:
        detail = f"{type(last_error).__name__}: {str(last_error)[:300]}"
    raise RuntimeError(
        f"process judge failed after 3 attempts; last_error={detail}"
    ) from last_error


def _score_process_live(
    *,
    llm: LLMClient,
    diagnosis: Diagnosis,
    root: RootCause,
    steps: list[ReasoningStep],
    model: str,
    temperature: float,
) -> ProcessScoreEvidence:
    chain, checkpoints = _render_gt_context(steps)
    samples = [
        _call_process_judge(
            llm=llm,
            diagnosis=diagnosis,
            root=root,
            chain=chain,
            checkpoints=checkpoints,
            model=model,
            temperature=temperature,
        )
    ]
    if round(samples[0].score * 100) in {0, 20, 50}:
        for _ in range(2):
            samples.append(
                _call_process_judge(
                    llm=llm,
                    diagnosis=diagnosis,
                    root=root,
                    chain=chain,
                    checkpoints=checkpoints,
                    model=model,
                    temperature=temperature,
                )
            )
    samples.sort(key=lambda item: item.score)
    chosen = samples[len(samples) // 2]
    chosen.samples = len(samples)
    chosen.judge_model = model
    chosen.score_formula = _process_formula(chosen)
    if len(samples) > 1:
        values = "/".join(str(round(item.score * 100)) for item in samples)
        chosen.score_formula += f"；敏感档 {len(samples)} 采样取中位: {values}"
    return chosen


def _replay_dimension_decisions(
    score_file: Path,
) -> tuple[FaultTypeScoreEvidence, EntityScoreEvidence, ProcessScoreEvidence]:
    """Load frozen LLM/entity decisions while recomputing the final score kernel."""
    payload = json.loads(score_file.read_text(encoding="utf-8"))
    details = payload.get("details") or {}
    return (
        FaultTypeScoreEvidence.model_validate(details.get("fault_type") or {}),
        EntityScoreEvidence.model_validate(details.get("entity") or {}),
        ProcessScoreEvidence.model_validate(details.get("process") or {}),
    )


def _score_process_replay(
    score_file: Path,
    *,
    expected_fault_type: str,
) -> ProcessScoreEvidence:
    payload = json.loads(score_file.read_text(encoding="utf-8"))
    details = payload.get("details") or {}
    process = details.get("process") or {}
    evidence = score_process_from_review_dimensions(
        process.get("review_dimensions") or [],
        expected_fault_type=expected_fault_type,
        reasoning=process.get("reason") or "",
    )
    evidence.score_formula = str(
        process.get("score_formula") or _process_formula(evidence)
    )
    evidence.samples = int(process.get("samples") or 1)
    evidence.judge_model = str(process.get("judge_model") or "")
    return evidence


def _score_process_dimension(
    *,
    replay_process_file: Path | None,
    fault: FaultTypeScoreEvidence,
    diagnosis: Diagnosis,
    root: RootCause,
    steps: list[ReasoningStep],
    llm: LLMClient,
    model: str,
    temperature: float,
) -> ProcessScoreEvidence:
    evidence = (
        _score_process_replay(replay_process_file, expected_fault_type=root.fault_id)
        if replay_process_file is not None
        else _score_process_live(
            llm=llm,
            diagnosis=diagnosis,
            root=root,
            steps=steps,
            model=model,
            temperature=temperature,
        )
    )
    if fault.score == 0 and evidence.score > 0.4:
        evidence.capped_from = evidence.score
        evidence.score = 0.4
        evidence.cap_reasons.append("定因未命中：推理链导向错误根因")
    return evidence
