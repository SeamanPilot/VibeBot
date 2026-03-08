"""Backtest metric calculations and Monte Carlo perturbation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class BacktestMetrics:
    """Core performance metrics for a strategy run."""

    expectancy: float
    win_rate: float
    average_win: float
    average_loss: float
    payoff_ratio: float
    profit_factor: float
    max_drawdown: float
    sharpe: float
    sortino: float
    exposure_time: float


def calculate_metrics(
    equity_curve: pd.Series, trade_pnls: list[float], exposure_time: float
) -> BacktestMetrics:
    """Compute standard trading performance metrics."""

    pnl = np.array(trade_pnls, dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    win_rate = float((pnl > 0).mean()) if len(pnl) else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    expectancy = float(pnl.mean()) if len(pnl) else 0.0
    payoff_ratio = float(abs(avg_win / avg_loss)) if avg_loss != 0 else 0.0
    profit_factor = (
        float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else 0.0
    )

    max_drawdown = _max_drawdown(equity_curve)
    returns = equity_curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = (
        float(np.sqrt(252) * returns.mean() / returns.std())
        if returns.std() and len(returns)
        else 0.0
    )
    downside = returns[returns < 0]
    sortino = (
        float(np.sqrt(252) * returns.mean() / downside.std())
        if downside.std() and len(downside)
        else 0.0
    )
    return BacktestMetrics(
        expectancy=expectancy,
        win_rate=win_rate,
        average_win=avg_win,
        average_loss=avg_loss,
        payoff_ratio=payoff_ratio,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown,
        sharpe=sharpe,
        sortino=sortino,
        exposure_time=exposure_time,
    )


def monte_carlo_equity(
    trade_pnls: list[float],
    num_paths: int = 250,
    seed: int = 7,
) -> pd.DataFrame:
    """Bootstrap trade sequence to estimate distribution of ending equity and drawdown."""

    if not trade_pnls:
        return pd.DataFrame(columns=["path_id", "final_pnl", "max_drawdown"])
    rng = np.random.default_rng(seed)
    base = np.asarray(trade_pnls, dtype=float)
    rows = []
    for path_id in range(num_paths):
        sampled = rng.choice(base, size=len(base), replace=True)
        curve = np.cumsum(sampled)
        rows.append(
            {
                "path_id": path_id,
                "final_pnl": float(curve[-1]),
                "max_drawdown": _max_drawdown(pd.Series(curve)),
            }
        )
    return pd.DataFrame(rows)


def _max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    running_max = series.cummax()
    drawdown = running_max - series
    return float(drawdown.max())
