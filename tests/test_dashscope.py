from __future__ import annotations

import pytest

from brise_rca_score.llm.base import LLMMessage
from brise_rca_score.llm.dashscope import DashScopeLLM


class RecordingDashScopeLLM(DashScopeLLM):
    def __init__(self) -> None:
        super().__init__(api_key="test-key")
        self.payload: dict[str, object] = {}

    def _request_with_retry(self, payload: dict[str, object]) -> dict[str, object]:
        self.payload = payload
        return {"choices": [{"message": {"content": "{}"}}]}


class TruncatedReasoningDashScopeLLM(DashScopeLLM):
    def __init__(self) -> None:
        super().__init__(api_key="test-key")
        self.payloads: list[dict[str, object]] = []

    def _request_with_retry(self, payload: dict[str, object]) -> dict[str, object]:
        self.payloads.append(payload)
        if len(self.payloads) == 1:
            return {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "content": "",
                            "reasoning_content": "hidden reasoning used the budget",
                        },
                    }
                ]
            }
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": '{"ok": true}'}}
            ]
        }


class TruncatedKimiK3DashScopeLLM(TruncatedReasoningDashScopeLLM):
    def __init__(self) -> None:
        super().__init__()
        self._model = "kimi/kimi-k3"


def test_max_tokens_uses_call_argument_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DASHSCOPE_MAX_TOKENS", raising=False)
    llm = RecordingDashScopeLLM()
    llm.chat([LLMMessage(role="user", content="test")], max_tokens=2048)
    assert llm.payload["max_tokens"] == 2048


def test_max_tokens_can_be_overridden_by_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DASHSCOPE_MAX_TOKENS", "16384")
    llm = RecordingDashScopeLLM()
    llm.chat([LLMMessage(role="user", content="test")], max_tokens=2048)
    assert llm.payload["max_tokens"] == 16384


def test_json_call_retries_without_thinking_after_hidden_reasoning_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DASHSCOPE_DISABLE_THINKING", "0")
    llm = TruncatedReasoningDashScopeLLM()
    result = llm.chat([LLMMessage(role="user", content="test")], response_format="json")
    assert result == '{"ok": true}'
    assert len(llm.payloads) == 2
    assert "enable_thinking" not in llm.payloads[0]
    assert llm.payloads[1]["enable_thinking"] is False


def test_kimi_k3_length_fallback_keeps_thinking_and_expands_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DASHSCOPE_DISABLE_THINKING", "0")
    llm = TruncatedKimiK3DashScopeLLM()
    result = llm.chat(
        [LLMMessage(role="user", content="test")],
        model="kimi/kimi-k3",
        max_tokens=4096,
        response_format="json",
    )
    assert result == '{"ok": true}'
    assert len(llm.payloads) == 2
    assert "enable_thinking" not in llm.payloads[1]
    assert llm.payloads[1]["max_tokens"] == 8192


def test_explicit_structured_retry_disables_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DASHSCOPE_DISABLE_THINKING", "0")
    llm = RecordingDashScopeLLM()
    result = llm.chat_structured_retry(
        [LLMMessage(role="user", content="test")], response_format="json"
    )
    assert result == "{}"
    assert llm.payload["enable_thinking"] is False


def test_kimi_k3_structured_retry_keeps_required_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DASHSCOPE_DISABLE_THINKING", "0")
    llm = RecordingDashScopeLLM()
    result = llm.chat_structured_retry(
        [LLMMessage(role="user", content="test")],
        model="kimi/kimi-k3",
        response_format="json",
    )
    assert result == "{}"
    assert "enable_thinking" not in llm.payload


def test_reasoning_effort_can_be_set_by_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DASHSCOPE_REASONING_EFFORT", "low")
    llm = RecordingDashScopeLLM()
    llm.chat([LLMMessage(role="user", content="test")])
    assert llm.payload["reasoning_effort"] == "low"


def test_reasoning_effort_is_omitted_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DASHSCOPE_REASONING_EFFORT", raising=False)
    llm = RecordingDashScopeLLM()
    llm.chat([LLMMessage(role="user", content="test")])
    assert "reasoning_effort" not in llm.payload


@pytest.mark.parametrize("value", ["0", "-1", "not-an-integer"])
def test_max_tokens_rejects_invalid_environment_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("DASHSCOPE_MAX_TOKENS", value)
    llm = RecordingDashScopeLLM()
    with pytest.raises(ValueError, match="DASHSCOPE_MAX_TOKENS"):
        llm.chat([LLMMessage(role="user", content="test")])


import pytest


@pytest.fixture(autouse=True)
def _configure_endpoint(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_ENDPOINT", "https://example.invalid")


def test_endpoint_must_be_explicit(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_ENDPOINT", raising=False)
    with pytest.raises(ValueError, match="DASHSCOPE_ENDPOINT"):
        DashScopeLLM(api_key="placeholder")
