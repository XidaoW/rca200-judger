"""Parity tests for structured Agent-chain fault scoring."""

from __future__ import annotations

from brise_rca_score.scoring.fault_type import (
    ReasoningStep,
    extract_agent_chain_and_conclusion,
    score_against_merged_chain,
    serialize_structured_agent_chain,
)


def test_extract_mermaid_chain_and_conclusion_keyword():
    text = (
        "## 分析\n"
        "服务调用图：\n"
        "```mermaid\nA --> B\nB --> C\n```\n"
        "## 结论\nFinal root cause is X due to Y\n"
        "(剩余内容)" * 5
    )
    chain, conclusion = extract_agent_chain_and_conclusion(text)
    assert "A --> B" in chain
    assert "B --> C" in chain
    assert conclusion.startswith("## 结论")
    assert len(conclusion) <= 500


def test_extract_no_keyword_falls_back_to_tail():
    """No conclusion keyword → falls back to last 500 chars."""
    text = "x" * 1000
    chain, conclusion = extract_agent_chain_and_conclusion(text)
    assert chain == ""
    assert len(conclusion) == 500


def test_extract_empty():
    assert extract_agent_chain_and_conclusion("") == ("", "")


# ---------------------------------------------------------------------------
# serialize_structured_agent_chain
# ---------------------------------------------------------------------------
def test_serialize_skips_empty_parts():
    nodes = [
        {"entity_name": "inventory", "detail": "线程池满", "fault_id": "F002"},
        {"entity_name": "cart", "detail": "", "fault_id": ""},  # detail+fid empty
        {"entity_name": "", "detail": "downstream", "fault_id": ""},  # only detail
    ]
    out = serialize_structured_agent_chain(nodes)
    assert out == "inventory | 线程池满 | F002\ncart\ndownstream"


def test_serialize_empty_input():
    assert serialize_structured_agent_chain([]) == ""
    assert serialize_structured_agent_chain(None) == ""


# ---------------------------------------------------------------------------
# score_against_merged_chain
# ---------------------------------------------------------------------------
def test_merged_chain_downstream():
    """GT chain [A→B], LLM virtual append [C]. Agent says C → hop=2 in merged
    chain (GT contributes 2 nodes, then LLM C at idx=2). LLM cap 0.50."""
    gt = [
        ReasoningStep(step=1, fault_id="F001-A"),
        ReasoningStep(step=2, fault_id="F002-B"),
    ]
    out = score_against_merged_chain(
        gt_steps=gt,
        llm_chain_path=[{"fault_id": "F003-C"}],
        chain_direction="downstream",
        actual_fault_types=["F003-C"],
    )
    assert out is not None
    # idx=2 → causal_2hop would be 0.65, but LLM cap is 0.50
    assert out.score == 0.50
    assert out.tier == "merged_2hop"
    assert out.best_actual == "F003-C"


def test_merged_chain_upstream():
    """LLM prepends; Agent says first node → hop=0 → score=1.0, but LLM cap 0.50."""
    gt = [ReasoningStep(step=1, fault_id="F010-Z")]
    out = score_against_merged_chain(
        gt_steps=gt,
        llm_chain_path=[{"fault_id": "F099-X"}],
        chain_direction="upstream",
        actual_fault_types=["F099-X"],
    )
    assert out is not None
    assert out.score == 0.50  # LLM cap dominates
    assert out.tier == "merged_0hop"


def test_merged_chain_gt_node_no_cap():
    """When the matched node is GT (not LLM), no cap applies."""
    gt = [
        ReasoningStep(step=1, fault_id="F001-A"),
        ReasoningStep(step=2, fault_id="F002-B"),
    ]
    out = score_against_merged_chain(
        gt_steps=gt,
        llm_chain_path=[{"fault_id": "F099-X"}],
        chain_direction="downstream",
        actual_fault_types=["F002-B"],  # GT node at idx=1
    )
    assert out is not None
    assert out.score == 0.85  # causal_1hop, no cap
    assert out.tier == "causal_1hop"


def test_merged_chain_no_llm_path_returns_none():
    out = score_against_merged_chain(
        gt_steps=[ReasoningStep(step=1, fault_id="F001-A")],
        llm_chain_path=[],
        chain_direction="downstream",
        actual_fault_types=["F001-A"],
    )
    assert out is None


def test_merged_chain_no_match_returns_none():
    out = score_against_merged_chain(
        gt_steps=[ReasoningStep(step=1, fault_id="F001-A")],
        llm_chain_path=[{"fault_id": "F999-Z"}],
        chain_direction="downstream",
        actual_fault_types=["F042-UNRELATED"],
    )
    assert out is None
