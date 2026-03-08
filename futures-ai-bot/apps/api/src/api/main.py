"""FastAPI entrypoint for health, TradingView webhook, and admin endpoints."""

from __future__ import annotations

from datetime import UTC, datetime

from common import configure_logging
from common.metadata import load_futures_metadata
from common.models import TradingViewWebhookPayload
from common.settings import get_settings
from fastapi import FastAPI, HTTPException

from api.state import build_state

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(title="Futures AI Bot API", version="0.1.0")
state = build_state()


@app.get("/health")
def health() -> dict[str, str | bool | int]:
    """Lightweight health check endpoint."""

    metadata = load_futures_metadata(settings.futures_metadata_path)
    return {
        "status": "ok",
        "timestamp": datetime.now(UTC).isoformat(),
        "paper_trading_enabled": settings.paper_trading_enabled,
        "live_broker_enabled": settings.live_broker_enabled,
        "configured_symbols": len(metadata.symbols),
    }


@app.post("/webhooks/tradingview")
def tradingview_webhook(payload: TradingViewWebhookPayload) -> dict:
    """Receive TradingView alerts and enqueue actions for paper execution."""

    if payload.secret != settings.tradingview_shared_secret:
        raise HTTPException(status_code=401, detail="Invalid shared secret")
    state.record_alert(payload)
    result = state.apply_webhook_action(payload)
    return {"status": "ok", "result": result}


@app.get("/admin/recent-alerts")
def recent_alerts(limit: int = 50) -> dict:
    """Return recent TradingView alert events."""

    return {"items": list(state.recent_alerts)[:limit]}


@app.get("/admin/recent-orders")
def recent_orders(limit: int = 50) -> dict:
    """Return recent order and fill state."""

    snapshot = state.broker.snapshot()
    orders = snapshot["open_orders"][:limit]
    fills = snapshot["fills"][:limit]
    return {"open_orders": orders, "recent_fills": fills}


@app.get("/admin/positions")
def positions() -> dict:
    """Return current paper positions."""

    snapshot = state.broker.snapshot()
    return {"positions": snapshot["positions"]}


@app.get("/admin/risk-state")
def risk_state() -> dict:
    """Return current risk engine state."""

    return state.risk.state()
