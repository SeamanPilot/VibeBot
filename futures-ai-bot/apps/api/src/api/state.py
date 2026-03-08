"""Application state for API webhook integration and paper execution."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from common.metadata import load_futures_metadata, metadata_by_root
from common.models import ActionType, NormalizedBar, TradingViewWebhookPayload
from common.settings import get_settings
from executor.models import BracketRequest, OrderRequest, OrderSide, OrderType
from executor.paper_broker import PaperBroker
from executor.risk import RiskConfig, RiskEngine


@dataclass(slots=True)
class AppState:
    """In-memory app state for alerts, orders, and risk snapshots."""

    broker: PaperBroker
    risk: RiskEngine
    recent_alerts: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=300))
    pending_actions: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=300))
    recent_predictions: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=300))

    def record_alert(self, payload: TradingViewWebhookPayload) -> None:
        self.recent_alerts.appendleft(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "event_type": payload.event_type,
                "symbol": payload.symbol,
                "timeframe": payload.timeframe,
                "action": payload.action.value,
                "confidence": payload.confidence,
                "strategy_id": payload.strategy_id,
            }
        )

    def enqueue_action(
        self, action: ActionType, payload: TradingViewWebhookPayload
    ) -> dict[str, Any]:
        item = {
            "queued_at": datetime.now(UTC).isoformat(),
            "symbol": payload.symbol,
            "action": action.value,
            "confidence": payload.confidence or 0.0,
        }
        self.pending_actions.appendleft(item)
        return item

    def apply_webhook_action(self, payload: TradingViewWebhookPayload) -> dict[str, Any]:
        """Apply TradingView action to the paper broker."""

        symbol = payload.symbol
        confidence = payload.confidence if payload.confidence is not None else 0.5
        position = self.broker.positions.get(symbol)
        current_qty = 0 if position is None else position.quantity
        decision = self.risk.evaluate(
            symbol=symbol,
            proposed_qty=max(abs(current_qty), 1),
            confidence=confidence,
            timestamp=payload.timestamp,
            data_age_seconds=0.0,
            latency_ms=10,
            open_positions_count=sum(1 for p in self.broker.positions.values() if p.quantity != 0),
        )
        action = payload.action
        queued = self.enqueue_action(action, payload)
        if not decision.allowed:
            queued["status"] = "rejected"
            queued["reasons"] = decision.reasons
            return queued

        if action == ActionType.LONG_ENTRY and current_qty == 0:
            price = payload.price or 0.0
            atr = max(abs(price) * 0.001, 0.5)
            self.broker.submit_bracket_order(
                BracketRequest(
                    symbol=symbol,
                    entry_side=OrderSide.BUY,
                    quantity=1,
                    entry_order_type=OrderType.MARKET,
                    stop_loss_price=price - atr if price else 0.0,
                    profit_target_price=price + (atr * 1.2) if price else 0.0,
                )
            )
        elif action == ActionType.SHORT_ENTRY and current_qty == 0:
            price = payload.price or 0.0
            atr = max(abs(price) * 0.001, 0.5)
            self.broker.submit_bracket_order(
                BracketRequest(
                    symbol=symbol,
                    entry_side=OrderSide.SELL,
                    quantity=1,
                    entry_order_type=OrderType.MARKET,
                    stop_loss_price=price + atr if price else 0.0,
                    profit_target_price=price - (atr * 1.2) if price else 0.0,
                )
            )
        elif action == ActionType.EXIT_LONG and current_qty > 0:
            self.broker.submit_order(
                OrderRequest(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=current_qty,
                    order_type=OrderType.MARKET,
                )
            )
        elif action == ActionType.EXIT_SHORT and current_qty < 0:
            self.broker.submit_order(
                OrderRequest(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=abs(current_qty),
                    order_type=OrderType.MARKET,
                )
            )
        elif action == ActionType.REDUCE and current_qty != 0:
            qty = max(abs(current_qty) // 2, 1)
            side = OrderSide.SELL if current_qty > 0 else OrderSide.BUY
            self.broker.submit_order(
                OrderRequest(symbol=symbol, side=side, quantity=qty, order_type=OrderType.MARKET)
            )
        elif action == ActionType.NO_TRADE:
            queued["status"] = "ignored"
            return queued

        if payload.price is not None:
            bar = NormalizedBar(
                timestamp=payload.timestamp,
                symbol_root=symbol,
                timeframe=payload.timeframe,
                open=payload.price,
                high=payload.price,
                low=payload.price,
                close=payload.price,
                volume=0.0,
                source="tradingview_webhook",
            )
            fills = self.broker.process_bar(bar)
            for fill in fills:
                if fill.realized_pnl_delta != 0:
                    self.risk.on_realized_trade(fill.realized_pnl_delta)
        queued["status"] = "accepted"
        return queued


def build_state() -> AppState:
    """Create application state from config and metadata."""

    settings = get_settings()
    metadata = metadata_by_root(load_futures_metadata(settings.futures_metadata_path))
    broker = PaperBroker(metadata)
    risk = RiskEngine(
        metadata,
        RiskConfig(
            max_daily_loss=settings.max_daily_loss,
            max_loss_streak=settings.max_loss_streak,
            max_concurrent_positions=settings.max_concurrent_positions,
            min_confidence=0.5,
        ),
    )
    return AppState(broker=broker, risk=risk)
