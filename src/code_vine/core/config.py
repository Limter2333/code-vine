from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


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


@lru_cache
def get_settings() -> Settings:
    return Settings()
