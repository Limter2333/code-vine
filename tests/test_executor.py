"""执行器测试：重点测「不抛出」和「is_error 语义」。"""

from anthropic.types import ToolUseBlock

from code_vine.agent.executor import run_all_tool_uses, run_tool_use
from code_vine.agent.tools import ToolRegistry, ToolSpec
from code_vine.agent.tools_demo import build_demo_registry


def _use(id_: str, name: str, input_: dict) -> ToolUseBlock:
    return ToolUseBlock.model_validate(
        {"type": "tool_use", "id": id_, "name": name, "input": input_}
    )


def test_ok_path() -> None:
    result = run_tool_use(_use("t1", "get_time", {}), build_demo_registry())
    assert result["tool_use_id"] == "t1"
    assert not result.get("is_error", False)
    assert isinstance(result["content"], str)


def test_unknown_tool_is_error_not_raise() -> None:
    result = run_tool_use(_use("t2", "hacked", {}), build_demo_registry())
    assert result["is_error"] is True
    assert "未知工具" in str(result["content"])


def test_broken_tool_is_error_not_raise() -> None:
    def boom(**_: str) -> str:
        raise RuntimeError("数据库连不上")

    reg = ToolRegistry().register(
        ToolSpec("boom", "测试用", {"type": "object", "properties": {}}, boom)
    )
    result = run_tool_use(_use("t3", "boom", {}), reg)
    assert result["is_error"] is True
    assert "RuntimeError" in str(result["content"])
    assert "数据库连不上" in str(result["content"])


def test_wrong_signature_is_error_not_raise() -> None:
    """模型给错参数形状 → TypeError → 变 is_error，而不是炸掉整个循环。"""
    result = run_tool_use(_use("t4", "calculator", {"wrong_key": "1"}), build_demo_registry())
    assert result["is_error"] is True


def test_run_all_collects_every_tool_use() -> None:
    """一条 assistant 消息里两个 tool_use → 两个结果块，顺序一致、id 对得上。"""
    blocks = [_use("a1", "get_time", {}), _use("a2", "calculator", {"expression": "1+1"})]
    results = run_all_tool_uses(blocks, build_demo_registry())
    assert [r["tool_use_id"] for r in results] == ["a1", "a2"]
    assert all(not r.get("is_error", False) for r in results)
