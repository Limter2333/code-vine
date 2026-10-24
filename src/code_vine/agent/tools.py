"""工具契约与注册表（纯逻辑，无 I/O）。

契约（name/description/input_schema）发给模型；fn 只留在本地 —— 分界线在这里。
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    """一个工具的对外契约。

    description 是**给模型读的文档**，不是给程序员读的注释 ——
    写清楚「什么时候该调我」，比写清楚「我怎么实现」重要十倍。
    """

    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema，模型按它生成参数
    fn: Callable[..., str]  # 实际执行体，返回字符串（模型读得懂的形态）

    def to_api(self) -> dict[str, Any]:
        """转成 messages.create(tools=[...]) 要的形状。fn 不出网。"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class ToolRegistry:
    """名字 → 契约 的查找表。≈ HandlerMapping：按路由找到处理器。"""

    _specs: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> "ToolRegistry":
        """注册并返回 self，支持链式调用。重名直接炸 —— 静默覆盖是生产事故。"""
        if spec.name in self._specs:
            raise ValueError(f"工具重名: {spec.name}")
        self._specs[spec.name] = spec
        return self

    def get(self, name: str) -> ToolSpec | None:
        """查不到返回 None —— 模型可能编出一个不存在的工具名，调用方要能处理。"""
        return self._specs.get(name)

    def api_tools(self) -> list[dict[str, Any]]:
        """给 messages.create(tools=...) 的参数。"""
        return [spec.to_api() for spec in self._specs.values()]

    def api_tools_names(self) -> list[str]:
        """已注册的工具名。主要用于给模型的错误提示。"""
        return sorted(self._specs)
