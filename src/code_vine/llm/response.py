"""响应处理：按 block.type 分支渲染 + 完成性校验。"""
from collections.abc import Sequence

from anthropic.types import ContentBlock, Message


class IncompleteResponseError(RuntimeError):
    """响应没到可信终点（截断/拒答等）。携带原始 message，供调用方记账。"""

    def __init__(self, message: Message) -> None:
        self.message = message
        self.stop_reason = message.stop_reason
        super().__init__(f"响应未可信完成: stop_reason={self.stop_reason}")


def ensure_complete(message: Message) -> None:
    """业务完成性校验。只信 stop_reason == end_turn。"""
    if message.stop_reason != "end_turn":
        raise IncompleteResponseError(message)


def render_content(content: Sequence[ContentBlock]) -> str:
    """content 是块列表不是字符串 —— 按 type 分支，未知类型也要有出口。"""
    parts: list[str] = []
    for block in content:
        if block.type == "text":
            parts.append(block.text)
        elif block.type == "thinking":
            parts.append(f"[thinking] {block.thinking}")
        else:
            parts.append(f"[未处理的块: {block.type}]")
    return "\n".join(parts)
