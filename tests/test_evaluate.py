#!/usr/bin/env python3
"""Metric tests: verify Sharpe/Sortino/Calmar/drawdown/profit-factor against
hand-computable synthetic equity curves and trade lists."""

from __future__ import annotations

import numpy as np

from backtest import BacktestResult, TradeRecord
from evaluate import compute_metrics


def _result(equity, returns, trades=None, timeframe="1h"):
    return BacktestResult(
        symbol="TEST",
        timeframe=timeframe,
        strategy_name="synthetic",
        period="train",
        equity_curve=np.asarray(equity, dtype=np.float64),
        returns=np.asarray(returns, dtype=np.float64),
        trades=trades or [],
        num_bars=len(equity),
    )


def _trade(pnl, pnl_pct=0.0):
    return TradeRecord(
        entry_time=None, exit_time=None, direction=1,
        entry_price=100.0, exit_price=101.0, size=1.0, leverage=1.0,
        pnl=pnl, pnl_pct=pnl_pct, funding_cost=0.0, fee_cost=0.0,
    )


def test_total_return_pct():
    eq = [10_000, 11_000, 12_000]
    m = compute_metrics(_result(eq, [0.0, 0.1, 0.0909]))
    assert abs(m["total_return_pct"] - 20.0) < 1e-6


def test_max_drawdown():
    # Peak 12000, trough 9000 → -25% drawdown.
    eq = [10_000, 12_000, 9_000, 10_500]
    m = compute_metrics(_result(eq, [0.0, 0.2, -0.25, 0.1667]))
    assert abs(m["max_drawdown_pct"] - (-25.0)) < 1e-6


def test_no_drawdown_is_zero():
    eq = [10_000, 10_500, 11_000]
    m = compute_metrics(_result(eq, [0.0, 0.05, 0.0476]))
    assert m["max_drawdown_pct"] == 0.0


def test_sharpe_positive_for_steady_gains():
    rets = np.full(200, 0.001)
    rets[0] = 0.0
    eq = 10_000 * np.cumprod(1 + rets)
    m = compute_metrics(_result(eq, rets))
    assert m["sharpe_ratio"] > 0


def test_sharpe_negative_for_steady_losses():
    rets = np.full(200, -0.001)
    rets[0] = 0.0
    eq = 10_000 * np.cumprod(1 + rets)
    m = compute_metrics(_result(eq, rets))
    assert m["sharpe_ratio"] < 0


def test_win_rate_and_profit_factor():
    trades = [_trade(100), _trade(100), _trade(100), _trade(-50)]  # 3 win / 1 loss
    eq = [10_000, 10_250]
    m = compute_metrics(_result(eq, [0.0, 0.025], trades=trades))
    assert m["num_trades"] == 4
    assert abs(m["win_rate"] - 75.0) < 1e-6
    # gross profit 300, gross loss 50 → PF = 6.0
    assert abs(m["profit_factor"] - 6.0) < 1e-6


def test_profit_factor_inf_when_no_losses():
    trades = [_trade(10), _trade(20)]
    m = compute_metrics(_result([10_000, 10_030], [0.0, 0.003], trades=trades))
    assert m["profit_factor"] == float("inf")


def test_zero_trades_metrics_are_safe():
    m = compute_metrics(_result([10_000, 10_000], [0.0, 0.0]))
    assert m["num_trades"] == 0
    assert m["win_rate"] == 0.0
    assert m["profit_factor"] == 0.0
    assert m["sharpe_ratio"] == 0.0


def test_calmar_uses_drawdown():
    rets = np.full(400, 0.001)
    rets[0] = 0.0
    rets[200] = -0.2  # inject a drawdown
    eq = 10_000 * np.cumprod(1 + rets)
    m = compute_metrics(_result(eq, rets))
    assert m["max_drawdown_pct"] < 0
    # calmar = annual_return / |max_dd|
    assert np.isfinite(m["calmar_ratio"])
