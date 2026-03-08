"""Streamlit operations dashboard for paper trading research workflows."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd
import streamlit as st


def fetch_json(url: str) -> dict | None:
    """Fetch JSON from an HTTP endpoint."""

    try:
        with urlopen(url, timeout=2) as response:  # noqa: S310 - controlled local endpoint
            payload = json.loads(response.read().decode("utf-8"))
            if isinstance(payload, dict):
                return payload
            return None
    except (URLError, TimeoutError, json.JSONDecodeError):
        return None


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV if present, else empty frame."""

    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_registry(path: Path) -> pd.DataFrame:
    """Read local model registry index for active/shadow status."""

    if not path.exists():
        return pd.DataFrame()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(payload.get("records", []))


st.set_page_config(page_title="Futures AI Bot", layout="wide")
st.title("Futures AI Bot Dashboard (Paper Mode)")

with st.sidebar:
    st.header("Controls")
    api_base = st.text_input("API Base URL", value="http://localhost:8000")
    paper_mode_toggle = st.toggle("Paper Trading Enabled", value=True, disabled=True)
    st.caption(f"Paper mode: {'ON' if paper_mode_toggle else 'OFF'}")

health = fetch_json(f"{api_base}/health")
risk = fetch_json(f"{api_base}/admin/risk-state")
positions = fetch_json(f"{api_base}/admin/positions")
orders = fetch_json(f"{api_base}/admin/recent-orders")
alerts = fetch_json(f"{api_base}/admin/recent-alerts")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("System Status")
    st.json(health or {"status": "offline"})
with col2:
    st.subheader("Risk State")
    st.json(risk or {"status": "unavailable"})
with col3:
    st.subheader("Positions")
    st.json(positions or {"positions": []})

st.subheader("Pending Orders and Recent Fills")
st.json(orders or {"open_orders": [], "recent_fills": []})

st.subheader("Recent Alerts")
st.json(alerts or {"items": []})

reports_dir = Path("reports")
predictions = load_csv(reports_dir / "recent_predictions.csv")
equity = load_csv(reports_dir / "equity_curve.csv")
by_symbol = load_csv(reports_dir / "by_symbol.csv")
drift_summary_path = reports_dir / "promotion_gate.json"

if not predictions.empty:
    st.subheader("Recent Predictions")
    st.dataframe(predictions.tail(100), use_container_width=True)

if not by_symbol.empty:
    st.subheader("Per-Symbol Stats")
    st.dataframe(by_symbol, use_container_width=True)

if not equity.empty:
    st.subheader("Equity Curve")
    st.line_chart(equity.set_index("timestamp")["equity"])
    st.subheader("Drawdown Curve")
    st.line_chart(equity.set_index("timestamp")["drawdown"])

if drift_summary_path.exists():
    st.subheader("Feature Drift Summary")
    st.json(json.loads(drift_summary_path.read_text(encoding="utf-8")))

registry = load_registry(Path("models/registry/index.json"))
if not registry.empty:
    st.subheader("Model Registry Snapshot")
    st.dataframe(
        registry[
            [
                "model_id",
                "version",
                "trained_at",
                "status",
                "featureset_version",
            ]
        ].head(20),
        use_container_width=True,
    )
