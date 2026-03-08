"""Event-driven backtesting engine for signal + paper broker evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from common.models import ActionType, FuturesSymbolMetadata, NormalizedBar
from executor.models import BracketRequest, OrderRequest, OrderSide, OrderType
from executor.paper_broker import PaperBroker
from executor.risk import RiskDecision, RiskEngine
from strategy.signal_engine import SignalContext, SignalEngine

from backtest.metrics import BacktestMetrics, calculate_metrics, monte_carlo_equity


@dataclass(slots=True)
class BacktestResult:
    """Structured backtest outputs for reporting and dashboarding."""

    metrics: BacktestMetrics
    equity_curve: pd.DataFrame
    decisions: pd.DataFrame
    fills: pd.DataFrame
    by_symbol: pd.DataFrame
    by_session: pd.DataFrame
    monte_carlo: pd.DataFrame


class EventDrivenBacktester:
    """Replay feature bars and execute signal actions in a paper broker."""

    def __init__(
        self,
        metadata_by_symbol: dict[str, FuturesSymbolMetadata],
        broker: PaperBroker,
        signal_engine: SignalEngine,
        risk_engine: RiskEngine,
    ) -> None:
        self.metadata_by_symbol = metadata_by_symbol
        self.broker = broker
        self.signal_engine = signal_engine
        self.risk_engine = risk_engine

    def run(self, frame: pd.DataFrame) -> BacktestResult:
        """Run event-driven simulation over a feature+prediction DataFrame."""

        required = {
            "timestamp",
            "symbol_root",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "probability_long",
            "ensemble_score",
            "regime",
            "confidence",
            "rolling_vol",
            "atr",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing backtest inputs: {missing}")

        bars = frame.sort_values(["timestamp", "symbol_root"]).reset_index(drop=True).copy()
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        decision_rows: list[dict[str, Any]] = []
        equity_rows: list[dict[str, Any]] = []
        exposed_count = 0

        for row in bars.itertuples(index=False):
            bar = NormalizedBar(
                timestamp=row.timestamp,
                symbol_root=row.symbol_root,
                timeframe="1m",
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                source="backtest",
            )

            fills = self.broker.process_bar(bar)
            for fill in fills:
                if fill.realized_pnl_delta != 0:
                    self.risk_engine.on_realized_trade(fill.realized_pnl_delta)

            position = self.broker.positions.get(row.symbol_root)
            current_qty = 0 if position is None else position.quantity
            if current_qty != 0:
                exposed_count += 1

            context = SignalContext(
                symbol=row.symbol_root,
                timestamp=row.timestamp,
                probability_long=float(row.probability_long),
                ensemble_score=float(row.ensemble_score),
                regime=int(row.regime),
                confidence=float(row.confidence),
                rolling_vol=float(row.rolling_vol) if pd.notna(row.rolling_vol) else 0.0,
                current_position_qty=current_qty,
                data_age_seconds=0.0,
            )
            action, reject_reasons = self.signal_engine.decide(context)
            risk_decision = RiskDecision(allowed=True, reasons=[])
            if action in {
                ActionType.LONG_ENTRY,
                ActionType.SHORT_ENTRY,
                ActionType.EXIT_LONG,
                ActionType.EXIT_SHORT,
                ActionType.REDUCE,
            }:
                risk_decision = self.risk_engine.evaluate(
                    symbol=row.symbol_root,
                    proposed_qty=max(abs(current_qty), 1),
                    confidence=context.confidence,
                    timestamp=row.timestamp,
                    data_age_seconds=context.data_age_seconds,
                    latency_ms=25,
                    open_positions_count=sum(
                        1 for p in self.broker.positions.values() if p.quantity != 0
                    ),
                )

            if risk_decision.allowed:
                self._apply_action(
                    action=action,
                    symbol=row.symbol_root,
                    close=float(row.close),
                    atr=float(row.atr),
                    position_qty=current_qty,
                )
            else:
                reject_reasons.extend(risk_decision.reasons)

            decision_rows.append(
                {
                    "timestamp": row.timestamp,
                    "symbol_root": row.symbol_root,
                    "action": action.value,
                    "reasons": "|".join(reject_reasons),
                    "probability_long": row.probability_long,
                    "ensemble_score": row.ensemble_score,
                    "regime": row.regime,
                    "confidence": row.confidence,
                }
            )
            equity_rows.append(
                {
                    "timestamp": row.timestamp,
                    "equity": self.broker.equity,
                    "realized_pnl": self.broker.realized_pnl,
                    "unrealized_pnl": self.broker.unrealized_pnl,
                    "drawdown": self.broker.max_drawdown,
                }
            )

        fills_df = pd.DataFrame([fill.model_dump(mode="json") for fill in self.broker.fills])
        if fills_df.empty:
            fills_df = pd.DataFrame(
                columns=[
                    "timestamp",
                    "order_id",
                    "symbol",
                    "side",
                    "quantity",
                    "price",
                    "commission",
                    "realized_pnl_delta",
                ]
            )
        fills_df["timestamp"] = pd.to_datetime(fills_df["timestamp"], utc=True, errors="coerce")
        trade_pnls = [float(x) for x in fills_df["realized_pnl_delta"].tolist() if float(x) != 0.0]
        exposure_time = exposed_count / max(len(bars), 1)

        equity_curve = pd.DataFrame(equity_rows).drop_duplicates("timestamp", keep="last")
        metrics = calculate_metrics(
            equity_curve=(
                equity_curve["equity"] if not equity_curve.empty else pd.Series(dtype=float)
            ),
            trade_pnls=trade_pnls,
            exposure_time=exposure_time,
        )
        by_symbol = _symbol_analytics(fills_df)
        by_session = _session_analytics(fills_df)
        monte_carlo = monte_carlo_equity(trade_pnls)
        decisions = pd.DataFrame(decision_rows)
        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_curve,
            decisions=decisions,
            fills=fills_df,
            by_symbol=by_symbol,
            by_session=by_session,
            monte_carlo=monte_carlo,
        )

    def _apply_action(
        self, action: ActionType, symbol: str, close: float, atr: float, position_qty: int
    ) -> None:
        if action == ActionType.LONG_ENTRY and position_qty == 0:
            self.broker.submit_bracket_order(
                BracketRequest(
                    symbol=symbol,
                    entry_side=OrderSide.BUY,
                    quantity=1,
                    entry_order_type=OrderType.MARKET,
                    stop_loss_price=close - max(atr, close * 0.001),
                    profit_target_price=close + max(atr * 1.2, close * 0.0012),
                )
            )
            return

        if action == ActionType.SHORT_ENTRY and position_qty == 0:
            self.broker.submit_bracket_order(
                BracketRequest(
                    symbol=symbol,
                    entry_side=OrderSide.SELL,
                    quantity=1,
                    entry_order_type=OrderType.MARKET,
                    stop_loss_price=close + max(atr, close * 0.001),
                    profit_target_price=close - max(atr * 1.2, close * 0.0012),
                )
            )
            return

        if action == ActionType.EXIT_LONG and position_qty > 0:
            self.broker.submit_order(
                OrderRequest(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=position_qty,
                    order_type=OrderType.MARKET,
                )
            )
            return

        if action == ActionType.EXIT_SHORT and position_qty < 0:
            self.broker.submit_order(
                OrderRequest(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=abs(position_qty),
                    order_type=OrderType.MARKET,
                )
            )
            return

        if action == ActionType.REDUCE and position_qty != 0:
            reduce_qty = max(abs(position_qty) // 2, 1)
            side = OrderSide.SELL if position_qty > 0 else OrderSide.BUY
            self.broker.submit_order(
                OrderRequest(
                    symbol=symbol, side=side, quantity=reduce_qty, order_type=OrderType.MARKET
                )
            )

    def save_reports(self, result: BacktestResult, reports_dir: Path) -> None:
        """Write CSV and HTML backtest reports."""

        reports_dir.mkdir(parents=True, exist_ok=True)
        result.equity_curve.to_csv(reports_dir / "equity_curve.csv", index=False)
        result.decisions.to_csv(reports_dir / "decisions.csv", index=False)
        result.fills.to_csv(reports_dir / "fills.csv", index=False)
        result.by_symbol.to_csv(reports_dir / "by_symbol.csv", index=False)
        result.by_session.to_csv(reports_dir / "by_session.csv", index=False)
        result.monte_carlo.to_csv(reports_dir / "monte_carlo.csv", index=False)

        html = f"""
<html><body>
<h1>Backtest Report</h1>
<h2>Core Metrics</h2>
<pre>{result.metrics}</pre>
<h2>By Symbol</h2>
{result.by_symbol.to_html(index=False)}
<h2>By Session</h2>
{result.by_session.to_html(index=False)}
</body></html>
"""
        (reports_dir / "backtest_report.html").write_text(html, encoding="utf-8")


def _symbol_analytics(fills: pd.DataFrame) -> pd.DataFrame:
    if fills.empty:
        return pd.DataFrame(columns=["symbol", "fill_count", "net_pnl", "avg_fill_price"])
    return (
        fills.groupby("symbol")
        .agg(
            fill_count=("fill_id", "count"),
            net_pnl=("realized_pnl_delta", "sum"),
            avg_fill_price=("price", "mean"),
        )
        .reset_index()
    )


def _session_analytics(fills: pd.DataFrame) -> pd.DataFrame:
    if fills.empty:
        return pd.DataFrame(columns=["session", "fills", "net_pnl"])
    ny = fills["timestamp"].dt.tz_convert("America/New_York")
    hour = ny.dt.hour
    session = pd.Series("overnight", index=fills.index)
    session[(hour >= 9) & (hour < 12)] = "us_morning"
    session[(hour >= 12) & (hour < 16)] = "us_afternoon"
    session[(hour >= 16) & (hour < 20)] = "post_close"
    tmp = fills.copy()
    tmp["session"] = session
    return (
        tmp.groupby("session")
        .agg(fills=("fill_id", "count"), net_pnl=("realized_pnl_delta", "sum"))
        .reset_index()
    )
