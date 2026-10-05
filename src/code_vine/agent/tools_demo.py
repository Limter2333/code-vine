"""示例工具：不碰网络、结果确定，方便反复实验。

工具入参是**模型生成的**，所以是不可信输入 —— calculator 特意不用 eval()。
"""

import ast
import operator
from collections.abc import Callable
from datetime import datetime

from code_vine.agent.tools import ToolRegistry, ToolSpec

_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY: dict[type[ast.unaryop], Callable[[float], float]] = {ast.USub: operator.neg}


def get_time() -> str:
    """当前日期时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculator(expression: str) -> str:
    """安全求值：只放行数字和 + - * / ( )。绝不 eval()。"""
    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        raise ValueError(f"不支持的表达式元素: {type(node).__name__}")

    tree = ast.parse(expression, mode="eval")
    return str(ev(tree.body))


GET_TIME = ToolSpec(
    name="get_time",
    description="获取当前的日期与时间。当用户问「现在几点」「今天几号」「星期几」时调用。没有参数。",
    input_schema={"type": "object", "properties": {}, "required": []},
    fn=get_time,
)

CALCULATOR = ToolSpec(
    name="calculator",
    description=(
        "对一个算术表达式求值，只支持数字、+ - * / 、括号和空格。"
        "任何需要精确计算的场合都该调用它，不要心算、不要猜。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "算术表达式，例如 '17 * 23 + 1'",
            }
        },
        "required": ["expression"],
    },
    fn=calculator,
)


def build_demo_registry() -> ToolRegistry:
    """示例注册表。生产里这里会替换成真实的业务工具。"""
    return ToolRegistry().register(GET_TIME).register(CALCULATOR)
