from collections.abc import Mapping, Sequence
from typing import Any

from anthropic.types import MessageParam
from pydantic import BaseModel


def dump_history(history: Sequence[MessageParam], *, max_text: int = 300) -> str:
    """逐条渲染消息。长文本截断，免得轨迹比回答还长。"""
    out: list[str] = []
    for i, msg in enumerate(history, 1):
        role = msg.get("role", "?")
        out.append(f"\n── [{i}] {role} {'─' * 40}")
        content = msg.get("content", "")
        if isinstance(content, str):
            out.append(_clip(content, max_text))
            continue
        # 块列表：按 type 分支渲染，未知类型也要有出口
        for block in content or []:
            out.append(_dump_block(block, max_text))
    return "\n".join(out)


def _as_mapping(block: object) -> Mapping[str, Any]:
    """块的归一化：history 里混着两种形状。

    - dict 块：我们自己 append 的 tool_result / 测试造的数据
    - pydantic 对象块：append_assistant 原样入史的 SDK 响应（ThinkingBlock 等）
    """
    if isinstance(block, Mapping):
        return block
    if isinstance(block, BaseModel):
        return block.model_dump()
    return {"type": "?", "raw": block}  # 真没见过的东西也不能崩


def _dump_block(block: object, max_text: int) -> str:
    b = _as_mapping(block)  # ← 全部修复就这一行的功夫
    kind = b.get("type", "?")
    if kind == "text":
        return f"  [text] {_clip(str(b.get('text', '')), max_text)}"
    if kind == "thinking":
        return f"  [thinking] {_clip(str(b.get('thinking', '')), max_text)}"
    if kind == "tool_use":
        return (
            f"  [tool_use] {b.get('name')}({b.get('input')})"
            f"   id={b.get('id')}"
        )
    if kind == "tool_result":
        err = "  ⚠ is_error" if b.get("is_error") else ""
        return (
            f"  [tool_result]{err} id={b.get('tool_use_id')}"
            f" → {_clip(str(b.get('content', '')), max_text)}"
        )
    return f"  [{kind}] {b}"  # 未知块也留个出口


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + f"…（共 {len(text)} 字）"