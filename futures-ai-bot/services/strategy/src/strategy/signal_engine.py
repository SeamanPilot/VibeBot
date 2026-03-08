"""Signal engine that combines model output, regimes, and runtime filters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

from common.models import ActionType, FuturesSymbolMetadata


@dataclass(slots=True)
class SignalConfig:
    """Signal threshold and filter settings."""

    min_probability_long: float = 0.58
    min_probability_short: float = 0.42
    min_confidence: float = 0.55
    max_rolling_vol: float = 0.03
    stale_data_seconds: float = 30.0
    require_higher_tf_confirmation: bool = False
    allow_cross_market_override: bool = True


@dataclass(slots=True)
class SignalContext:
    """Per-bar decision inputs."""

    symbol: str
    timestamp: datetime
    probability_long: float
    ensemble_score: float
    regime: int
    confidence: float
    rolling_vol: float
    current_position_qty: int
    data_age_seconds: float
    higher_tf_confirmed: bool = True
    cross_market_filter_passed: bool = True


class SignalEngine:
    """Compute trade actions from model outputs and policy filters."""

    def __init__(
        self,
        metadata_by_symbol: dict[str, FuturesSymbolMetadata],
        config: SignalConfig | None = None,
    ) -> None:
        self.metadata_by_symbol = metadata_by_symbol
        self.config = config or SignalConfig()

    def decide(self, context: SignalContext) -> tuple[ActionType, list[str]]:
        """Return action and reject reasons (empty when action accepted)."""

        reasons: list[str] = []
        cfg = self.config
        metadata = self.metadata_by_symbol.get(context.symbol)
        if metadata is None:
            return ActionType.NO_TRADE, ["unknown_symbol"]
        if context.data_age_seconds > cfg.stale_data_seconds:
            return ActionType.NO_TRADE, ["stale_data"]
        if context.confidence < cfg.min_confidence:
            return ActionType.NO_TRADE, ["low_confidence"]
        if context.rolling_vol > cfg.max_rolling_vol:
            return ActionType.NO_TRADE, ["volatility_filter"]
        if cfg.require_higher_tf_confirmation and not context.higher_tf_confirmed:
            return ActionType.NO_TRADE, ["higher_tf_reject"]
        if cfg.allow_cross_market_override and not context.cross_market_filter_passed:
            return ActionType.NO_TRADE, ["cross_market_reject"]
        if not self._is_allowed_window(metadata, context.timestamp):
            return ActionType.NO_TRADE, ["outside_allowed_window"]
        if self._is_restricted_window(metadata, context.timestamp):
            return ActionType.NO_TRADE, ["restricted_window"]

        if context.current_position_qty == 0:
            if (
                context.probability_long >= cfg.min_probability_long
                and context.ensemble_score > 0.0
            ):
                return ActionType.LONG_ENTRY, reasons
            if (
                context.probability_long <= cfg.min_probability_short
                and context.ensemble_score < 0.0
            ):
                return ActionType.SHORT_ENTRY, reasons
            return ActionType.NO_TRADE, reasons

        if context.current_position_qty > 0:
            if context.probability_long <= 0.50 or context.ensemble_score < -0.25:
                return ActionType.EXIT_LONG, reasons
            if context.probability_long < cfg.min_probability_long:
                return ActionType.REDUCE, reasons
            return ActionType.HOLD, reasons

        if context.probability_long >= 0.50 or context.ensemble_score > 0.25:
            return ActionType.EXIT_SHORT, reasons
        if context.probability_long > cfg.min_probability_short:
            return ActionType.REDUCE, reasons
        return ActionType.HOLD, reasons

    def _is_allowed_window(self, symbol: FuturesSymbolMetadata, ts: datetime) -> bool:
        if not symbol.allowed_trade_windows:
            return True
        return any(
            self._is_within(window.start, window.end, window.tz, ts)
            for window in symbol.allowed_trade_windows
        )

    def _is_restricted_window(self, symbol: FuturesSymbolMetadata, ts: datetime) -> bool:
        return any(
            self._is_within(window.start, window.end, window.tz, ts)
            for window in symbol.restricted_windows
        )

    def _is_within(self, start_text: str, end_text: str, tz_name: str, ts: datetime) -> bool:
        local = ts.astimezone(ZoneInfo(tz_name)).time()
        start = _parse_hhmm(start_text)
        end = _parse_hhmm(end_text)
        if start <= end:
            return start <= local <= end
        return local >= start or local <= end


def _parse_hhmm(text: str) -> time:
    hh, mm = text.split(":")
    return time(hour=int(hh), minute=int(mm))
