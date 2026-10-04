import pytest

from code_vine.llm.cost import PRICES, Price, estimate_cost


def test_estimate_cost_formula() -> None:
    # 不依赖真实价目：用登记制塞一个假价格，验公式本身
    # 1M 输入 + 1M 输出，单价 4/20 → 4 + 20 = 24
    PRICES["test-model"] = Price(4.0, 20.0)
    assert estimate_cost(1_000_000, 1_000_000, "test-model") == 24.0
    # 500K 输入 + 250K 输出，单价 1/5 → 0.5 + 1.25 = 1.75
    PRICES["test-cheap"] = Price(1.0, 5.0)
    assert estimate_cost(500_000, 250_000, "test-cheap") == pytest.approx(1.75)


def test_unknown_model_raises() -> None:
    with pytest.raises(KeyError):
        estimate_cost(500_000, 500_000, "test-model")


def test_price_table_registered_for_real_models() -> None:
    # 这条测试故意红着：价格表还是占位 0 就过不了 —— 逼你去查真实价目填进来
    for model, price in PRICES.items():
        print(f"{model}: in={price.input_per_mtok} out={price.output_per_mtok}")
        assert price.input_per_mtok > 0, f"{model} 输入单价没登记真实值"
        assert price.output_per_mtok > 0, f"{model} 输出单价没登记真实值"
