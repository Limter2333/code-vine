from code_vine.agent.trace import dump_history


def test_dump_renders_all_known_block_types() -> None:
    history = [
        {"role": "user", "content": "现在几点？"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "用户要时间"},
                {"type": "text", "text": "我来查。"},
                {"type": "tool_use", "id": "tu_1", "name": "get_time", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "2026-10-06 09:12:33"}
            ],
        },
    ]
    out = dump_history(history)
    assert out.count("── [") == 3  # 三条消息
    assert "[tool_use] get_time" in out
    assert "[tool_result]" in out and "tu_1" in out
    assert "[thinking]" in out


def test_dump_marks_is_error() -> None:
    history = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "x", "content": "炸了", "is_error": True}
            ],
        }
    ]
    assert "⚠ is_error" in dump_history(history)


def test_dump_clips_long_text() -> None:
    history = [{"role": "user", "content": "长" * 500}]
    out = dump_history(history, max_text=50)
    assert "共 500 字" in out
    assert "长" * 500 not in out


def test_dump_unknown_block_has_exit() -> None:
    """未知块不能让渲染器崩 —— agent 会不断遇到没见过的东西。"""
    history = [{"role": "assistant", "content": [{"type": "some_future_block", "x": 1}]}]
    assert "[some_future_block]" in dump_history(history)
