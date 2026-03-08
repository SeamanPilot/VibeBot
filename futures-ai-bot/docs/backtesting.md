# Backtesting

## Engine
Event-driven backtester routes model decisions through the same paper broker and risk engine used by webhook execution.

## Outputs
- `reports/equity_curve.csv`
- `reports/fills.csv`
- `reports/decisions.csv`
- `reports/by_symbol.csv`
- `reports/by_session.csv`
- `reports/monte_carlo.csv`
- `reports/backtest_report.html`
- `reports/metrics_summary.csv`

## Metrics
- Expectancy
- Win rate
- Average win/loss
- Payoff ratio
- Profit factor
- Max drawdown
- Sharpe
- Sortino
- Exposure time

## Validation Pattern
Use walk-forward training/validation/test splits from `training.dataset` to reduce leakage and keep sequence integrity.
