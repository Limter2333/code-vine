"""会话内核：历史维护 + 单轮编排。API 无状态，历史完全由客户端持有。"""

from collections.abc import Sequence

from anthropic import Anthropic
from anthropic.types import ContentBlock, Message, MessageParam

from code_vine.core.config import Effort
from code_vine.llm.response import ensure_complete


def append_user(history: list[MessageParam], text: str) -> None:
    history.append({"role": "user", "content": text})


def append_assistant(history: list[MessageParam], content: Sequence[ContentBlock]) -> None:
    """原样追加块列表（含 thinking 块）。只抽 text 会在多轮时静默丢信息。"""
    # SDK 的响应块类型和请求块参数是两套类（≈ DTO vs VO），用 cast 过桥
    from typing import cast

    history.append(cast(MessageParam, {"role": "assistant", "content": list(content)}))


def count_input_tokens(
    client: Anthropic, history: list[MessageParam], *, system: str, model: str
) -> int:
    """发请求前预估输入侧 token（不含将要生成的输出）。"""
    result = client.messages.count_tokens(model=model, system=system, messages=history)
    return result.input_tokens


def run_turn(
    client: Anthropic,
    history: list[MessageParam],
    user_text: str,
    *,
    system: str,
    model: str,
    max_tokens: int,
    effort: Effort,
) -> Message:
    """一轮对话：追加 user → 调 API → 校验 → 追加 assistant → 返回。

    失败路径：IncompleteResponseError 时 user 悬在历史里，assistant 未追加 ——
    半截回答绝不冒充成功。
    """
    append_user(history, user_text)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        output_config={"effort": effort},  # 思考深度，≈ 资源旋钮
        thinking={"type": "adaptive", "display": "summarized"},  # 让 thinking 块可见
        messages=history,
    )
    ensure_complete(response)  # 先查 stop_reason，再信任 content
    append_assistant(history, response.content)
    return response
