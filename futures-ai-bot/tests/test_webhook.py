from datetime import UTC, datetime

from api.main import app
from fastapi.testclient import TestClient


def test_tradingview_webhook_accepts_valid_payload() -> None:
    client = TestClient(app)
    payload = {
        "secret": "change-me",
        "event_type": "entry",
        "symbol": "MES",
        "timeframe": "1m",
        "action": "LONG_ENTRY",
        "timestamp": datetime(2026, 1, 1, 14, 30, tzinfo=UTC).isoformat(),
        "price": 5100.5,
        "confidence": 0.7,
    }
    response = client.post("/webhooks/tradingview", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["result"]["status"] in {"accepted", "ignored", "rejected"}
