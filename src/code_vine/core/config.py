from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# 思考深度档位。用 Literal 而不是 str —— 非法值在启动时就报错，≈ Java 的 enum
Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Settings(BaseSettings):
    """应用配置。字段默认值兜底，环境变量 / .env 覆盖。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "code-vine"
    debug: bool = False
    api_prefix: str = "/api/v1"

    # ===== LLM（Anthropic API）=====
    anthropic_api_key: str = ""            # .env: ANTHROPIC_API_KEY
    anthropic_base_url: str | None = None  # 网关/中转用，直连留空
    llm_model: str = "qwen3.8-flash"       # 学习期测试模型（便宜档，反复试错成本低）
    llm_max_tokens: int = 16000            # 单次输出上限（≠上下文窗口）
    llm_effort: Effort = "high"            # 思考深度：low…max
    llm_proxy: str | None = None           # 如 socks5h://127.0.0.1:10808


@lru_cache
def get_settings() -> Settings:
    return Settings()
