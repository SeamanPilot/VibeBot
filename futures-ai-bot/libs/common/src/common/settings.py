"""Application settings and typed configuration loaders."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Environment-driven application settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="dev", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    paper_trading_enabled: bool = Field(default=True, alias="PAPER_TRADING_ENABLED")
    live_broker_enabled: bool = Field(default=False, alias="LIVE_BROKER_ENABLED")

    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    model_registry_dir: Path = Field(default=Path("models/registry"), alias="MODEL_REGISTRY_DIR")
    futures_metadata_path: Path = Field(
        default=Path("config/futures_metadata.yaml"), alias="FUTURES_METADATA_PATH"
    )

    tradingview_shared_secret: str = Field(default="change-me", alias="TRADINGVIEW_SHARED_SECRET")

    max_daily_loss: float = Field(default=1000.0, alias="MAX_DAILY_LOSS")
    max_loss_streak: int = Field(default=3, alias="MAX_LOSS_STREAK")
    max_concurrent_positions: int = Field(default=4, alias="MAX_CONCURRENT_POSITIONS")
    order_stale_seconds: int = Field(default=30, alias="ORDER_STALE_SECONDS")

    api_host: str = Field(default="0.0.0.0", alias="API_HOST")  # noqa: S104
    api_port: int = Field(default=8000, alias="API_PORT")


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return a cached settings instance."""

    return AppSettings()
