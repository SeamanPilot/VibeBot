[README.md](https://github.com/user-attachments/files/25828464/README.md)
# Futures AI Bot

Production-style Python 3.12 monorepo for futures research and paper trading focused on short-horizon scalp workflows.

## Safety Defaults
- Paper trading enabled by default.
- Live broker integration disabled by default.
- No martingale/grid/revenge/averaging-down logic.
- No auto-promotion of models to production.
- No profit guarantees.

## Repo Layout
- `apps/api`: FastAPI service (`/health`, TradingView webhook, admin views)
- `apps/dashboard`: Streamlit operations dashboard
- `services/executor`: Paper broker + risk engine
- `services/strategy`: Feature engineering + signal engine
- `services/backtest`: Event-driven backtesting + reports
- `services/training`: Ingestion, labels, dataset, training, registry, drift checks
- `libs/common`: Shared settings, models, metadata, logging, data utils

## Quickstart
```bash
python -m pip install -U pip
python -m pip install -e .[dev]
copy .env.example .env
python -m training.synthetic_data
python -m training.train train
python -m backtest.run
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
streamlit run apps/dashboard/app.py
```

## Developer Commands
```bash
make lint
make typecheck
make test
make demo
```

## Model Registry Commands
```bash
python -m training.train list
python -m training.train promote --model-id <id> --version <version>
python -m training.train rollback --model-id <id> --version <version>
```

## TradingView Assets
- `tradingview/indicator_alerts.pine`
- `tradingview/strategy_alerts.pine`
- `tradingview/payload_templates.json`

## Disclaimer
This repository is for research and paper simulation only by default. Real trading carries significant risk.
