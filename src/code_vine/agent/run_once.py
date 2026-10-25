import sys

from anthropic.types import MessageParam

from code_vine.agent.budget import AgentRun
from code_vine.agent.loop import run_agent
from code_vine.agent.tools_demo import build_demo_registry
from code_vine.agent.trace import dump_history
from code_vine.core.config import get_settings
from code_vine.llm.client import get_client
from code_vine.llm.conversation import append_user


def run_once(question: str, *, max_steps: int = 10) -> AgentRun:
    settings = get_settings()
    history: list[MessageParam] = []
    append_user(history, question)

    run = run_agent(
        get_client(),
        history,
        system="你是 code-vine 的助手。需要计算、查时间或看文件时使用工具，不要心算。",
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        effort=settings.llm_effort,
        registry=build_demo_registry(),
        max_steps=max_steps,
    )

    print(dump_history(history))
    print(f"\n[账本] {run.summary(settings.llm_model)}")
    return run


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "现在几点？另外帮我算一下 17 * 23 等于多少。"
    run_once(q)
