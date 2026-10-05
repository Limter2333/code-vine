"""工具契约的纯逻辑测试 —— 不打网络，跑得飞快。"""

import pytest

from code_vine.agent.tools_demo import build_demo_registry, calculator


def test_registry_to_api_omits_fn() -> None:
    """fn 绝不能出现在发给模型的契约里 —— 泄漏实现细节，还浪费 token。"""
    tools = build_demo_registry().api_tools()
    names = {t["name"] for t in tools}
    assert names == {"get_time", "calculator"}
    for t in tools:
        assert set(t) == {"name", "description", "input_schema"}


def test_registry_rejects_duplicate_name() -> None:
    from code_vine.agent.tools import ToolRegistry
    from code_vine.agent.tools_demo import CALCULATOR

    reg = ToolRegistry().register(CALCULATOR)
    with pytest.raises(ValueError, match="重名"):
        reg.register(CALCULATOR)


def test_unknown_tool_returns_none() -> None:

    assert build_demo_registry().get("CALCULATOR") is None


@pytest.mark.parametrize(
    ("expr", "expect"),
    [("17*23+1", "392.0"), ("(1+2)*3", "9.0"), ("-5+8", "3.0")],
)
def test_calculator(expr: str, expect: str) -> None:
    assert calculator(expr) == expect


def test_calculator_rejects_code_injection() -> None:
    """模型给的参数是不可信输入 —— 白名单 AST，进不来任何函数调用。"""
    with pytest.raises(ValueError):
        calculator("__import__('os').system('dir')")


def test_calculator_rejects_name_reference() -> None:
    with pytest.raises(ValueError):
        calculator("open('/etc/passwd')")
