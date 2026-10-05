import pytest

from anthropic.types import MessageParam

from code_vine.core.config import Settings, get_settings
from code_vine.llm.client import get_client
from code_vine.llm.conversation import count_input_tokens, run_turn
from code_vine.llm.cost import estimate_cost
from code_vine.llm.response import render_content

pytestmark = pytest.mark.integration

# 成本护栏: 集成测试是来验接线的, 不是来产出内容的。给小的 max_tokens。
LIVE_MAX_TOKENS = 1024


@pytest.fixture
def settings() -> Settings:
    s = get_settings()
    if not s.anthropic_api_key:
        pytest.skip("没配 ANTHROPIC_API_KEY, 跳过真实调用")
    return s


# def test_live_single_call_shape(settings: Settings) -> None:
#     """报文形状 - 这是假 client 永远造不出来的信息。"""
#     client = get_client()
#     message = client.messages.create(
#         model=settings.llm_model,
#         max_tokens=LIVE_MAX_TOKENS,
#         system="你是一个测试用的助手。",
#         messages=[{"role": "user", "content": "只回一个字: 好"}],
#     )

#     # 只断言不变量, 不断言措辞 - 模型输出是非确定性的
#     assert message.stop_reason == "end_turn"
#     assert message.usage.input_tokens > 0
#     assert message.usage.output_tokens > 0
#     assert any(block.type == "text" for block in message.content)

#     cost = estimate_cost(
#         message.usage.input_tokens, message.usage.output_tokens, settings.llm_model
#     )
#     print(
#         f"\n[usage] in={message.usage.input_tokens} out={message.usage.output_tokens}"
#         f" cost={cost:.6f}"
#     )
#     print(f"[blocks] {[block.type for block in message.content]}")


# def test_live_gateway_accepts_thinking_and_effort(settings: Settings) -> None:
#     """网关认不认 thinking / output_config。不认就会 400, 这条就红。"""
#     client = get_client()
#     message = client.messages.create(
#         model=settings.llm_model,
#         max_tokens=LIVE_MAX_TOKENS,
#         thinking={"type": "adaptive", "display": "summarized"},
#         output_config={"effort": settings.llm_effort},
#         messages=[{"role": "user", "content": "3 的 7 次方是多少? 先算再答。"}],
#     )

#     assert message.stop_reason == "end_turn"
#     assert render_content(message.content).strip() != ""

#     # 这条的产出是"观察"不是"断言": 块类型是网关给的, 不是你代码的对错
#     print(f"\n[blocks] {[block.type for block in message.content]}")
#     print(f"[rendered] {render_content(message.content)[:120]}")


# def test_live_input_tokens_grow_with_history(settings: Settings) -> None:
#     """用真接口证明无状态: 第 2 轮的输入含第 1 轮的全量历史。"""
#     client = get_client()
#     system = "你是助手。"
#     history: list[MessageParam] = []
#     args = {
#         "system": system,
#         "model": settings.llm_model,
#         "max_tokens": LIVE_MAX_TOKENS,
#         "effort": "low",
#     }

#     first = run_turn(client, history, "记住暗号: vine-77", **args)
#     second = run_turn(client, history, "暗号是什么?", **args)

#     assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
#     # 客户端重发了 system + 全量历史 + 新问题, 所以第 2 轮输入必然更大
#     assert second.usage.input_tokens > first.usage.input_tokens

#     print(f"\n[轮1] in={first.usage.input_tokens}  [轮2] in={second.usage.input_tokens}")
#     print(f"[轮2 回答] {render_content(second.content)[:100]}")


# def test_live_count_tokens_available(settings: Settings) -> None:
#     """发请求前的预检能不能用。"""
#     client = get_client()
#     short = count_input_tokens(
#         client, [{"role": "user", "content": "你好"}], system="s", model=settings.llm_model
#     )
#     long = count_input_tokens(
#         client, [{"role": "user", "content": "你好" * 200}], system="s", model=settings.llm_model
#     )

#     assert 0 < short < long
#     print(f"\n[预估] short={short} long={long}")

def test_multi_conversation(settings: Settings) -> None:
    client = get_client()
    system = "你是助手。"
    history: list[MessageParam] = []
    args = {
            "system": system,
            "model": settings.llm_model,
            "max_tokens": LIVE_MAX_TOKENS,
            "effort": "low",
    }
    q = run_turn(client, history, "你是技能开发小助手小五", **args)
    a = run_turn(client, history, "告诉我你是谁", **args)
    print(f"\n[a = ] {render_content(a.content)[:100]}")