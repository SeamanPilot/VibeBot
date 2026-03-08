"""Risk controls for paper execution and signal gating."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

from common.models import FuturesSymbolMetadata


@dataclass(slots=True)
class RiskConfig:
    """Risk limits and guardrails."""

    hard_kill_switch: bool = False
    max_daily_loss: float = 1000.0
    max_loss_streak: int = 3
    max_concurrent_positions: int = 4
    max_position_size_by_symbol: dict[str, int] = field(default_factory=dict)
    symbol_enabled: dict[str, bool] = field(default_factory=dict)
    stale_data_seconds: int = 30
    max_latency_ms: int = 2000
    min_confidence: float = 0.55
    flatten_all: bool = False
    flatten_end_of_session: bool = False


@dataclass(slots=True)
class RiskDecision:
    """Result of risk gate evaluation."""

    allowed: bool
    reasons: list[str]


class RiskEngine:
    """Evaluate and track runtime risk state for order/signal requests."""

    def __init__(
        self, metadata_by_symbol: dict[str, FuturesSymbolMetadata], config: RiskConfig | None = None
    ) -> None:
        self.metadata_by_symbol = metadata_by_symbol
        self.config = config or RiskConfig()
        self.daily_realized_pnl: float = 0.0
        self.loss_streak: int = 0
        self.last_reset_day: datetime = datetime.now(UTC)
        self.kill_switch_triggered: bool = False

    def evaluate(
        self,
        symbol: str,
        proposed_qty: int,
        confidence: float,
        timestamp: datetime,
        data_age_seconds: float,
        latency_ms: int,
        open_positions_count: int,
    ) -> RiskDecision:
        """Evaluate whether a new trade action is allowed."""

        self._reset_if_new_day(timestamp)
        reasons: list[str] = []
        cfg = self.config

        if cfg.hard_kill_switch or self.kill_switch_triggered:
            reasons.append("hard_kill_switch")
        if cfg.flatten_all:
            reasons.append("flatten_all_active")
        if self.daily_realized_pnl <= -abs(cfg.max_daily_loss):
            reasons.append("max_daily_loss_exceeded")
            self.kill_switch_triggered = True
        if self.loss_streak >= cfg.max_loss_streak:
            reasons.append("max_loss_streak_exceeded")
        if open_positions_count >= cfg.max_concurrent_positions:
            reasons.append("max_concurrent_positions")
        if data_age_seconds > cfg.stale_data_seconds:
            reasons.append("stale_data")
        if latency_ms > cfg.max_latency_ms:
            reasons.append("abnormal_latency")
        if confidence < cfg.min_confidence:
            reasons.append("low_confidence")
        if cfg.symbol_enabled and not cfg.symbol_enabled.get(symbol, True):
            reasons.append("symbol_disabled")

        max_size = cfg.max_position_size_by_symbol.get(symbol)
        if max_size is not None and proposed_qty > max_size:
            reasons.append("position_size_limit")

        if self._is_in_restricted_window(symbol=symbol, timestamp=timestamp):
            reasons.append("restricted_window")

        return RiskDecision(allowed=len(reasons) == 0, reasons=reasons)

    def on_realized_trade(self, pnl_after_fees: float) -> None:
        """Update daily risk counters after each closed-trade PnL event."""

        self.daily_realized_pnl += pnl_after_fees
        if pnl_after_fees < 0:
            self.loss_streak += 1
        elif pnl_after_fees > 0:
            self.loss_streak = 0

    def state(self) -> dict:
        """Return serializable risk state."""

        return {
            "daily_realized_pnl": self.daily_realized_pnl,
            "loss_streak": self.loss_streak,
            "kill_switch_triggered": self.kill_switch_triggered,
            "config": self.config.__dict__,
        }

    def _reset_if_new_day(self, timestamp: datetime) -> None:
        now = timestamp.astimezone(UTC)
        if now.date() != self.last_reset_day.date():
            self.daily_realized_pnl = 0.0
            self.loss_streak = 0
            self.last_reset_day = now
            self.kill_switch_triggered = False

    def _is_in_restricted_window(self, symbol: str, timestamp: datetime) -> bool:
        metadata = self.metadata_by_symbol.get(symbol)
        if metadata is None:
            return False
        for window in metadata.restricted_windows:
            zone = ZoneInfo(window.tz)
            local = timestamp.astimezone(zone)
            start = _parse_time(window.start)
            end = _parse_time(window.end)
            if _within_time_window(local.time(), start, end):
                return True
        return False


def _parse_time(value: str) -> time:
    hh, mm = value.split(":")
    return time(hour=int(hh), minute=int(mm))


def _within_time_window(current: time, start: time, end: time) -> bool:
    if start <= end:
        return start <= current <= end
    return current >= start or current <= end
