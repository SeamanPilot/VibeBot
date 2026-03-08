# Risk Controls

## Hard Rules
- Paper trading only by default.
- Live broker integration disabled by default.
- Trading is rejected on risk violations.

## Implemented Controls
- Hard kill switch.
- Max daily loss.
- Max loss streak.
- Max concurrent positions.
- Per-symbol enable/disable.
- Max position size by symbol.
- Stale-data rejection.
- Abnormal-latency rejection.
- Restricted-window rejection.
- Confidence-threshold rejection.
- Flatten-all control.
- Optional end-of-session flatten toggle.

## Execution Risk Modeling
- Per-symbol tick-size/tick-value PnL math.
- Configurable slippage and commissions.
- Partial fill simulation.
- Bracket and OCO handling.
