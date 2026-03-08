# Architecture

## System Modules
- `libs/common` provides typed settings, metadata schemas, normalized bar schema, and JSON logging.
- `services/training` ingests data, builds features/labels, trains baseline models, stores artifacts in a local registry, and runs drift/promotion gates.
- `services/strategy` computes short-horizon features and action decisions.
- `services/executor` simulates execution through an event-driven paper broker plus explicit risk controls.
- `services/backtest` replays model signals through the paper broker and writes analytics reports.
- `apps/api` receives TradingView webhooks and exposes admin state.
- `apps/dashboard` visualizes runtime and report artifacts.

## Data Flow
1. Ingest CSV/parquet or synthetic bars.
2. Normalize to canonical schema.
3. Build leakage-safe features and labels.
4. Train calibrated baseline + regime model.
5. Register as `shadow`.
6. Run backtests and shadow evaluation.
7. Promote manually when thresholds pass.
8. Use webhook/API + paper broker for simulated execution.

## Safety Controls
- Paper-only default execution path.
- Hard kill-switch and flatten controls.
- Confidence, stale-data, and restricted-window filters.
- Explicit model status lifecycle (`shadow`, `production`, `archived`).
