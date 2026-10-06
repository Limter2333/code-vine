"""账本测试：纯函数，最快的一批。"""

import pytest

from code_vine.agent.budget import AgentLoopError, AgentRun, check_steps


def test_check_steps_allows_within_budget() -> None:
    check_steps(10, 10)  # 不抛


def test_check_steps_blocks_over_budget() -> None:
    with pytest.raises(AgentLoopError, match="最大步数"):
        check_steps(11, 10)


def test_charge_accumulates() -> None:
    """两次调用的 usage 要累加，不是覆盖 —— 账本的语义是「运行总账」。"""
    from anthropic.types import Message

    def msg(inp: int, out: int, cache: int = 0) -> Message:
        return Message.model_validate(
            {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "x"}],
                "model": "qwen3.8-flash",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": inp,
                    "output_tokens": out,
                    "cache_read_input_tokens": cache,
                    "cache_creation_input_tokens": 0,
                },
            }
        )

    run = AgentRun()
    run.charge(msg(100, 10))
    run.charge(msg(200, 20, cache=50))
    assert (run.input_tokens, run.output_tokens, run.cache_read_tokens) == (300, 30, 50)


def test_cost_is_positive_and_additive() -> None:
    run = AgentRun(input_tokens=1_000_000, output_tokens=1_000_000)
    assert run.cost("qwen3.8-flash") == pytest.approx(0.2 + 2.7)
