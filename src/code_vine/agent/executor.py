"""执行 tool_use 块 → 生成 tool_result 块。

铁律：本模块的任何路径都不抛异常。工具失败也返回 is_error=True 的结果块 ——
模型需要知道失败了才能改口，吞掉异常就是骗它。
"""

from collections.abc import Sequence
from typing import Any, cast

from anthropic.types import ContentBlock, ToolResultBlockParam, ToolUseBlock

from code_vine.agent.tools import ToolRegistry


def _ok(tool_use_id: str, text: str) -> ToolResultBlockParam:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": text}


def _err(tool_use_id: str, text: str) -> ToolResultBlockParam:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": text, "is_error": True}


def run_tool_use(block: ToolUseBlock, registry: ToolRegistry) -> ToolResultBlockParam:
    """执行一个 tool_use 块，返回对应的 tool_result 块。**永不抛出。**"""
    spec = registry.get(block.name)
    if spec is None:
        # 模型会编出不存在的工具名。告诉它，让它自己纠偏，而不是我们硬中断。
        return _err(block.id, f"未知工具: {block.name!r}。可用工具: {registry.api_tools_names()}")

    try:
        # block.input 是 SDK 解析好的 dict。类型标注是 dict[str, Any] ——
        # 值的结构没人保证，展开成关键字参数前不做校验，靠 fn 的签名兜底。
        args: dict[str, Any] = dict(block.input)
        return _ok(block.id, spec.fn(**args))
    except Exception as exc:
        # 带上异常类型名，模型能读出「参数给错了」还是「工具坏了」
        return _err(block.id, f"{type(exc).__name__}: {exc}")


def run_all_tool_uses(
    blocks: Sequence[ContentBlock], registry: ToolRegistry
) -> list[ToolResultBlockParam]:
    """跑完一条 assistant 消息里的**全部** tool_use 块。

    返回的列表必须在**同一条** user 消息里回传 —— 拆开发会驯化模型少开并行。
    """
    results: list[ToolResultBlockParam] = []
    for block in blocks:
        if block.type == "tool_use":
            results.append(run_tool_use(cast(ToolUseBlock, block), registry))
    return results
