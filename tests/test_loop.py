from typing import Any

import pytest
from anthropic.types import Message, MessageParam

from code_vine.agent.budget import AgentLoopError, AgentRun
from code_vine.agent.loop import run_agent
from code_vine.agent.tools_demo import build_demo_registry
from code_vine.llm.response import IncompleteResponseError


def make_message(
    content: list[dict[str, Any]], stop_reason: str, inp: int = 100, out: int = 10
) -> Message:
    """按线格式造一条假响应。≈ Mockito 的 thenReturn(...)。"""
    return Message.model_validate(
        {
            "id": "msg_fake",
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": "qwen3.8-flash",
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": inp,
                "output_tokens": out,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        }
    )


class FakeMessages:
    def __init__(self, script: list[Message]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Message:
        self.calls.append(kwargs)
        return self.script.pop(0)


class FakeClient:
    """鸭子类型：只要长得像 Anthropic 就行。Python 不要求 implements。"""

    def __init__(self, script: list[Message]) -> None:
        self.messages = FakeMessages(script)


TOOL_USE_CONTENT = [
    {"type": "text", "text": "我来算一下。"},
    {"type": "tool_use", "id": "tu_1", "name": "calculator", "input": {"expression": "17*23"}},
]
END_CONTENT = [{"type": "text", "text": "结果是 391。"}]


def _run(script: list[Message]) -> tuple[AgentRun, list[MessageParam], FakeClient]:
    client = FakeClient(script)
    history: list[MessageParam] = [{"role": "user", "content": "17*23 等于多少？"}]
    run = run_agent(
        client,  # type: ignore[arg-type]  # 鸭子类型：FakeClient 足够跑
        history,
        system="你是计算器助手",
        model="qwen3.8-flash",
        max_tokens=1000,
        effort="low",
        registry=build_demo_registry(),
        max_steps=5,
    )
    return run, history, client


def test_loop_one_tool_round_then_end() -> None:
    """最经典的两步：tool_use → end_turn。"""
    run, history, client = _run(
        [make_message(TOOL_USE_CONTENT, "tool_use"), make_message(END_CONTENT, "end_turn")]
    )
    assert run.steps == 2
    assert run.final_text == "结果是 391。"
    # 历史序列：user → assistant(tool_use) → user(tool_result) → assistant(text)
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    # tool_result 在一条 user 消息里
    assert history[2]["content"][0]["type"] == "tool_result"
    assert history[2]["content"][0]["tool_use_id"] == "tu_1"
    # 每次请求都带上了 tools
    assert all("tools" in c for c in client.messages.calls)


def test_parallel_tool_uses_share_one_user_message() -> None:
    """铁律一的回归测试：两个 tool_use → 一条 user 消息里两个 tool_result。"""
    two_uses = [
        {"type": "text", "text": "两件事一起做。"},
        {"type": "tool_use", "id": "tu_a", "name": "get_time", "input": {}},
        {"type": "tool_use", "id": "tu_b", "name": "calculator", "input": {"expression": "1+1"}},
    ]
    run, history, _ = _run(
        [make_message(two_uses, "tool_use"), make_message(END_CONTENT, "end_turn")]
    )
    tool_msgs = [m for m in history if m["role"] == "user" and isinstance(m["content"], list)]
    assert len(tool_msgs) == 1  # 只有一条 user 消息装结果
    assert [r["tool_use_id"] for r in tool_msgs[0]["content"]] == ["tu_a", "tu_b"]
    assert run.steps == 2


def test_failed_tool_becomes_is_error_and_loop_continues() -> None:
    """铁律二的回归测试：工具炸了 → is_error → 循环继续 → 模型自己收场。"""
    bad = [
        {"type": "text", "text": "算一下。"},
        {"type": "tool_use", "id": "tu_x", "name": "calculator", "input": {"expression": "1/0"}},
    ]
    run, history, _ = _run([make_message(bad, "tool_use"), make_message(END_CONTENT, "end_turn")])
    result = history[2]["content"][0]
    assert result["is_error"] is True
    assert "ZeroDivisionError" in str(result["content"])
    assert run.steps == 2  # 没有卡死


def test_max_tokens_raises_incomplete() -> None:
    """真 client 测不出这个 —— 你没法命令模型「输出超长」。只能编剧本。"""
    with pytest.raises(IncompleteResponseError):
        _run([make_message([{"type": "text", "text": "半截"}], "max_tokens")])


def test_refusal_raises() -> None:
    """同上：真 client 测不出拒答，假 client 一条剧本搞定。"""
    with pytest.raises(AgentLoopError, match="拒绝"):
        _run([make_message([{"type": "text", "text": "我不能做这个。"}], "refusal")])


def test_pause_turn_resumes() -> None:
    """pause_turn 是合法中间态：原样追加 content 后继续，不能当成错误。"""
    paused = [{"type": "text", "text": "任务较长，先暂停。"}]
    run, history, _ = _run(
        [make_message(paused, "pause_turn"), make_message(END_CONTENT, "end_turn")]
    )
    assert run.steps == 2
    assert run.final_text == "结果是 391。"


def test_max_steps_guard_trips() -> None:
    """永远说 tool_use → 必须被护栏拦下，不能烧钱烧到天亮。"""
    endless = [make_message(TOOL_USE_CONTENT, "tool_use") for _ in range(9)]
    client = FakeClient(endless)
    history: list[MessageParam] = [{"role": "user", "content": "永远循环"}]
    with pytest.raises(AgentLoopError, match="最大步数"):
        run_agent(
            client,  # type: ignore[arg-type]
            history,
            system="s",
            model="qwen3.8-flash",
            max_tokens=100,
            effort="low",
            registry=build_demo_registry(),
            max_steps=3,
        )
    assert len(client.messages.calls) == 3  # 精确 3 次，不多不少
