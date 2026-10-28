"""rcaeval.llm.stub — canned-response LLMClient for tests + offline scoring.

Two use cases:
  1. Fixture tests: inject a pre-recorded brise review_dimensions[] so we can
     drive rcaeval.process end-to-end without hitting any LLM.
  2. Offline runs: when user supplies a structured_diagnosis bundle that
     already contains review_dimensions, skip the LLM entirely.
"""

from __future__ import annotations

from .base import LLMClient, LLMMessage


class StubLLM(LLMClient):
    """Returns a canned response regardless of input."""

    def __init__(self, canned_response: str):
        self._response = canned_response
        self.calls: list[list[LLMMessage]] = []

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: str = "text",
    ) -> str:
        self.calls.append(messages)
        return self._response
