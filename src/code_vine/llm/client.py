"""Anthropic 客户端工厂：配置 → 客户端，一处接线。"""
from anthropic import Anthropic, DefaultHttpxClient

from code_vine.core.config import Settings, get_settings


def build_client(settings: Settings) -> Anthropic:
    """按 Settings 构造客户端。代理走 llm_proxy —— 我们的真实约束。"""
    if settings.llm_proxy:
        return Anthropic(
            api_key=settings.anthropic_api_key,
            base_url=settings.anthropic_base_url,
            http_client=DefaultHttpxClient(proxy=settings.llm_proxy),
        )
    return Anthropic(
        api_key=settings.anthropic_api_key,
        base_url=settings.anthropic_base_url,
    )


def get_client() -> Anthropic:
    """模块级单例入口（≈ @Bean 的单例语义；将来接 FastAPI 就包成 Depends）。"""
    return build_client(get_settings())
