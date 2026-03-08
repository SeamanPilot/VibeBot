# TradingView Setup

## 1. Configure Webhook URL
Use your API endpoint:
- `http://<host>:8000/webhooks/tradingview`

## 2. Shared Secret
Set `TRADINGVIEW_SHARED_SECRET` in `.env` and in Pine input.

## 3. Pine Scripts
- Indicator alerts: `tradingview/indicator_alerts.pine`
- Strategy alerts: `tradingview/strategy_alerts.pine`

## 4. Alert Payloads
Reference `tradingview/payload_templates.json` for `signal_only`, `entry`, `exit`, and `heartbeat` payloads.

## 5. Verify
- `GET /health`
- `GET /admin/recent-alerts`
- `GET /admin/recent-orders`
- `GET /admin/positions`
- `GET /admin/risk-state`
