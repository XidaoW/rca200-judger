"""Public scorer API orchestration."""

from __future__ import annotations

from pathlib import Path

from .contracts import OfflineScore
from .inputs import load_case_inputs
from .llm.base import LLMClient
from .multi_root_engine import _score_multi_root_case
from .process_runtime import _replay_dimension_decisions, _score_process_dimension
from .resolution import _reasoning_steps
from .single_root import (
    _build_offline_result,
    _score_entity_dimension,
    _score_fault_dimension,
)
from .trajectory import load_process_trajectory


def score_case(
    *,
    gt_file: Path,
    diagnosis_file: Path | None = None,
    agent_output_file: Path | None = None,
    llm: LLMClient,
    model: str = "qwen3.7-plus",
    temperature: float = 0.0,
    credibility: str = "RELIABLE",
    replay_process_file: Path | None = None,
    replay_decision_file: Path | None = None,
    topology_file: Path | None = None,
    trajectory_file: Path | None = None,
    vote_count: int = 3,
) -> OfflineScore:
    """Score one existing case using only local files plus the configured LLM."""
    bundle, diagnosis, digest, scored_output_source = load_case_inputs(
        gt_file,
        diagnosis_file,
        agent_output_file,
    )
    process_context = load_process_trajectory(trajectory_file)
    if process_context is not None:
        diagnosis = diagnosis.model_copy(
            update={
                "analysis_output": process_context.analysis_output,
                "tool_calls_summary": process_context.tool_calls_summary,
            }
        )
    roots = bundle.ground_truth.outcome.root_causes
    if replay_decision_file is not None:
        fault, entity, process = _replay_dimension_decisions(replay_decision_file)
        return _build_offline_result(
            diagnosis=diagnosis,
            gt_file=gt_file,
            scored_output_source=scored_output_source,
            scoring_source="decision_replay",
            digest=digest,
            model=model,
            credibility=credibility,
            fault=fault,
            entity=entity,
            process=process,
        )
    if len(roots) != 1:
        return _score_multi_root_case(
            bundle=bundle,
            diagnosis=diagnosis,
            gt_file=gt_file,
            scored_output_source=scored_output_source,
            digest=digest,
            llm=llm,
            model=model,
            temperature=temperature,
            credibility=credibility,
            topology_file=topology_file,
            replay_process_file=replay_process_file,
            vote_count=vote_count,
        )
    root = roots[0]
    steps = _reasoning_steps(bundle.ground_truth)
    fault = _score_fault_dimension(
        diagnosis=diagnosis,
        root=root,
        steps=steps,
        llm=llm,
        model=model,
        temperature=temperature,
        vote_count=vote_count,
    )
    entity = _score_entity_dimension(root, diagnosis, topology_file)
    process = _score_process_dimension(
        replay_process_file=replay_process_file,
        fault=fault,
        diagnosis=diagnosis,
        root=root,
        steps=steps,
        llm=llm,
        model=model,
        temperature=temperature,
    )
    return _build_offline_result(
        diagnosis=diagnosis,
        gt_file=gt_file,
        scored_output_source=scored_output_source,
        scoring_source=(
            "process_replay" if replay_process_file is not None else "dashscope"
        ),
        digest=digest,
        model=model,
        credibility=credibility,
        fault=fault,
        entity=entity,
        process=process,
    )
