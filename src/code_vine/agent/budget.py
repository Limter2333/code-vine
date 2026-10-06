"""步数护栏 + token/成本账本（纯逻辑，无 I/O）。"""

from dataclasses import dataclass, field

from anthropic.types import Message

from code_vine.llm.cost import estimate_cost


class AgentLoopError(RuntimeError):
    """循环本身出了问题（超过步数、未知 stop_reason 等）—— 不是工具的错。"""


@dataclass
class AgentRun:
    """一次 agent 运行的账本。所有轮次的数字累加在这里。"""

    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0  # 命中缓存的输入 token —— 只读，不重复计费
    tool_calls: list[str] = field(default_factory=list)  # 按调用顺序记工具名
    final_text: str = ""

    def charge(self, message: Message) -> None:
        """把一轮的 usage 记进账本。cache_read 可能是 None。"""
        self.input_tokens += message.usage.input_tokens
        self.output_tokens += message.usage.output_tokens
        self.cache_read_tokens += message.usage.cache_read_input_tokens or 0

    def cost(self, model: str) -> float:
        """本次运行的估算成本。单位同 cost.PRICES。"""
        return estimate_cost(self.input_tokens, self.output_tokens, model)

    def summary(self, model: str) -> str:
        return (
            f"步数={self.steps} 工具调用={self.tool_calls} "
            f"输入={self.input_tokens} 输出={self.output_tokens} "
            f"缓存读={self.cache_read_tokens} 估算成本={self.cost(model):.6f}"
        )


def check_steps(step: int, max_steps: int) -> None:
    """步数护栏。超过就抛 AgentLoopError —— 宁可中断，不烧钱。"""
    if step > max_steps:
        raise AgentLoopError(
            f"超过最大步数 {max_steps}，疑似死循环。"
            f"常见原因: 工具一直失败 / 描述写得模型找不到北 / 任务本身需要更多步数"
        )
