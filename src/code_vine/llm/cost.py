"""成本换算（纯函数，无 I/O）。价格单位：以你网关的价目为准（¥ 或 $）/ 百万 token（MTok）。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input_per_mtok: float  # 输入单价
    output_per_mtok: float  # 输出单价


# 价格表是登记制：未知模型抛 KeyError，不猜价。
# ⚠ 验收前必须把网关控制台的真实价目抄进来（含单位），0 只是待填占位 —— 有专门的测试盯着这件事。
PRICES: dict[str, Price] = {
    "qwen3.8-flash": Price(0.2, 2.7),  # TODO: 查网关价目填真实单价
    "qwen3.8-max": Price(12.0, 36.0),  # TODO: 同上（对照用）
}


def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """一次调用的计费成本（单位 = PRICES 的单位）。"""
    price = PRICES[model]
    return (input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / 1_000_000
