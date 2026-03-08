"""Typed shared models and enums used across services."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator


class ActionType(StrEnum):
    """Supported signal actions."""

    NO_TRADE = "NO_TRADE"
    LONG_ENTRY = "LONG_ENTRY"
    SHORT_ENTRY = "SHORT_ENTRY"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    REDUCE = "REDUCE"
    HOLD = "HOLD"


class Window(BaseModel):
    """Simple session/trading window definition."""

    name: str
    start: str
    end: str
    tz: str = "America/New_York"


class SymbolMappings(BaseModel):
    """Cross-provider symbol mapping definitions."""

    tradingview: str
    broker: str
    data_vendor: str


class FuturesSymbolMetadata(BaseModel):
    """Futures symbol metadata."""

    root: str
    description: str
    asset_class: str
    tick_size: float = Field(gt=0)
    tick_value: float = Field(gt=0)
    multiplier: float = Field(gt=0)
    currency: str = "USD"
    default_commission_per_contract: float = Field(ge=0)
    default_slippage_ticks: float = Field(ge=0)
    session_windows: list[Window]
    allowed_trade_windows: list[Window]
    restricted_windows: list[Window]
    mappings: SymbolMappings


class NormalizedBar(BaseModel):
    """Normalized OHLCV bar schema."""

    timestamp: datetime
    symbol_root: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float = Field(ge=0)
    source: str = "unknown"

    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, value: str) -> str:
        allowed = {"1s", "5s", "15s", "1m", "5m"}
        if value not in allowed:
            raise ValueError(f"Unsupported timeframe: {value}")
        return value

    @model_validator(mode="after")
    def validate_ohlc(self) -> NormalizedBar:
        if self.high < max(self.open, self.close):
            raise ValueError("high must be >= max(open, close)")
        if self.low > min(self.open, self.close):
            raise ValueError("low must be <= min(open, close)")
        if self.high < self.low:
            raise ValueError("high must be >= low")
        return self


class TradingViewWebhookPayload(BaseModel):
    """Incoming TradingView webhook payload."""

    secret: str
    event_type: str
    symbol: str
    timeframe: str
    action: ActionType
    timestamp: datetime
    price: float | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    strategy_id: str | None = None
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class EventRecord(BaseModel):
    """Persisted event record."""

    event_id: str
    event_type: str
    created_at: datetime
    payload: dict
