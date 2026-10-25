from typing import cast

from anthropic import Anthropic
from anthropic.types import MessageParam, ToolParam, ToolUseBlock

from code_vine.agent.budget import AgentLoopError, AgentRun, check_steps
from code_vine.agent.executor import run_all_tool_uses
from code_vine.agent.tools import ToolRegistry
from code_vine.core.config import Effort
from code_vine.llm.conversation import append_assistant
from code_vine.llm.response import IncompleteResponseError, render_content


def run_agent(
    client: Anthropic,
    history: list[MessageParam],
    *,
    system: str,
    model: str,
    max_tokens: int,
    effort: Effort,
    registry: ToolRegistry,
    max_steps: int = 10,
) -> AgentRun:
   
    run = AgentRun()
    step = 0

    while True:
        step += 1
        check_steps(step, max_steps)  # 护栏写在脸上：宁可中断，不烧钱
        run.steps = step

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            output_config={"effort": effort},
            thinking={"type": "adaptive", "display": "summarized"},
            tools=cast(list[ToolParam], registry.api_tools()),  # ← 与 run_turn 唯一的新增参数
            messages=history,
        )
        run.charge(response)  # 先记账，再处理 —— 被打断也要留下数字

        # ① 原样追加 assistant 块（含 tool_use 块）。复用阶段 1 的实现，一行没改。
        append_assistant(history, response.content)

        # ② 按 stop_reason 分支 —— 循环的唯一驱动力
        if response.stop_reason == "tool_use":
            # 记轨迹（工具名）—— 先记名字，再执行，失败了也知道炸在哪个工具上
            for b in response.content:
                if b.type == "tool_use":
                    run.tool_calls.append(cast(ToolUseBlock, b).name)
            # 执行本轮全部 tool_use，结果合成**一条** user 消息（铁律一）
            results = run_all_tool_uses(response.content, registry)
            history.append(cast(MessageParam, {"role": "user", "content": results}))
            continue

        if response.stop_reason in ("end_turn", "stop_sequence"):
            run.final_text = render_content(response.content)
            return run

        if response.stop_reason == "max_tokens":
            # 半截回答绝不冒充成功 —— 阶段 1 的原则原样继承
            raise IncompleteResponseError(response)

        if response.stop_reason == "refusal":
            raise AgentLoopError(f"模型拒绝回答: {render_content(response.content)}")

        if response.stop_reason == "pause_turn":
            # 长任务被暂停：content 已追加，直接再发一轮接着跑
            continue

        raise AgentLoopError(f"未知 stop_reason: {response.stop_reason!r}")
