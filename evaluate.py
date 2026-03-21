#!/usr/bin/env python3
"""
evaluate.py - Strategy Evaluation Metrics
==========================================
IMMUTABLE FILE - Do not modify during research experiments.

Computes comprehensive performance metrics for backtest results.
All metrics follow standard quantitative finance conventions.

Usage:
    from evaluate import compute_metrics, print_metrics
    metrics = compute_metrics(result)
    print_metrics(metrics)
"""

import numpy as np
import pandas as pd
from typing import Optional

from backtest import BacktestResult


# =============================================================================
# Annualization Factors
# =============================================================================

TIMEFRAME_BARS_PER_YEAR = {
    "1m": 365.25 * 24 * 60,
    "5m": 365.25 * 24 * 12,
    "15m": 365.25 * 24 * 4,
    "30m": 365.25 * 24 * 2,
    "1h": 365.25 * 24,
    "4h": 365.25 * 6,
    "6h": 365.25 * 4,
    "12h": 365.25 * 2,
    "1d": 365.25,
    "1w": 52.18,
}


def bars_per_year(timeframe: str) -> float:
    """Get annualization factor for a timeframe."""
    return TIMEFRAME_BARS_PER_YEAR.get(timeframe, 365.25 * 24)  # default to 1h


# =============================================================================
# Core Metrics
# =============================================================================

def compute_metrics(
    result: BacktestResult,
    risk_free_rate: float = 0.05,
) -> dict:
    """
    Compute comprehensive performance metrics.

    Returns dict with keys:
        - total_return_pct
        - annual_return_pct
        - sharpe_ratio
        - sortino_ratio
        - calmar_ratio
        - max_drawdown_pct
        - max_drawdown_duration_bars
        - win_rate
        - profit_factor
        - avg_trade_pnl
        - avg_win / avg_loss
        - num_trades
        - total_fees
        - total_funding_cost
        - exposure_pct (% of time in market)
        - volatility_annual_pct
        - skewness
        - kurtosis
        - best_trade_pct / worst_trade_pct
        - avg_bars_in_trade
        - expectancy
    """
    equity = result.equity_curve
    returns = result.returns
    trades = result.trades
    n = len(equity)
    tf = result.timeframe

    bpy = bars_per_year(tf)
    rf_per_bar = risk_free_rate / bpy

    metrics = {}

    # --- Return metrics ---
    initial = equity[0] if equity[0] > 0 else 1.0
    final = equity[-1]
    total_return = (final - initial) / initial
    metrics["total_return_pct"] = total_return * 100

    # Annualized return (CAGR)
    n_years = n / bpy
    if n_years > 0 and final > 0:
        metrics["annual_return_pct"] = ((final / initial) ** (1 / n_years) - 1) * 100
    else:
        metrics["annual_return_pct"] = 0.0

    # --- Risk metrics ---
    # Non-zero returns for volatility calculation
    non_zero_returns = returns[returns != 0]

    if len(non_zero_returns) > 1:
        vol = np.std(returns) * np.sqrt(bpy)
        metrics["volatility_annual_pct"] = vol * 100

        # Sharpe ratio
        excess_returns = returns - rf_per_bar
        mean_excess = np.mean(excess_returns)
        std_returns = np.std(returns)
        metrics["sharpe_ratio"] = (mean_excess / std_returns * np.sqrt(bpy)) if std_returns > 0 else 0.0

        # Sortino ratio (downside deviation only)
        downside = returns[returns < 0]
        if len(downside) > 0:
            downside_std = np.std(downside)
            metrics["sortino_ratio"] = (mean_excess / downside_std * np.sqrt(bpy)) if downside_std > 0 else 0.0
        else:
            metrics["sortino_ratio"] = float("inf") if mean_excess > 0 else 0.0

        # Higher moments
        metrics["skewness"] = float(pd.Series(returns).skew())
        metrics["kurtosis"] = float(pd.Series(returns).kurtosis())
    else:
        metrics["volatility_annual_pct"] = 0.0
        metrics["sharpe_ratio"] = 0.0
        metrics["sortino_ratio"] = 0.0
        metrics["skewness"] = 0.0
        metrics["kurtosis"] = 0.0

    # --- Drawdown metrics ---
    peak = np.maximum.accumulate(equity)
    drawdown = np.where(peak > 0, (equity - peak) / peak, 0.0)
    metrics["max_drawdown_pct"] = float(np.min(drawdown) * 100)

    # Max drawdown duration (bars)
    in_drawdown = drawdown < 0
    if np.any(in_drawdown):
        dd_starts = np.where(np.diff(in_drawdown.astype(int)) == 1)[0]
        dd_ends = np.where(np.diff(in_drawdown.astype(int)) == -1)[0]
        if len(dd_ends) == 0:
            dd_ends = np.array([n - 1])
        if len(dd_starts) > 0:
            max_dd_dur = 0
            for s in dd_starts:
                ends_after = dd_ends[dd_ends > s]
                if len(ends_after) > 0:
                    max_dd_dur = max(max_dd_dur, ends_after[0] - s)
                else:
                    max_dd_dur = max(max_dd_dur, n - 1 - s)
            metrics["max_drawdown_duration_bars"] = max_dd_dur
        else:
            metrics["max_drawdown_duration_bars"] = 0
    else:
        metrics["max_drawdown_duration_bars"] = 0

    # Calmar ratio
    if metrics["max_drawdown_pct"] < 0:
        metrics["calmar_ratio"] = metrics["annual_return_pct"] / abs(metrics["max_drawdown_pct"])
    else:
        metrics["calmar_ratio"] = float("inf") if metrics["annual_return_pct"] > 0 else 0.0

    # --- Trade metrics ---
    n_trades = len(trades)
    metrics["num_trades"] = n_trades

    if n_trades > 0:
        pnls = np.array([t.pnl for t in trades])
        pnl_pcts = np.array([t.pnl_pct for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        metrics["win_rate"] = len(wins) / n_trades * 100
        metrics["avg_trade_pnl"] = float(np.mean(pnls))
        metrics["avg_trade_pnl_pct"] = float(np.mean(pnl_pcts) * 100)
        metrics["avg_win"] = float(np.mean(wins)) if len(wins) > 0 else 0.0
        metrics["avg_loss"] = float(np.mean(losses)) if len(losses) > 0 else 0.0

        # Profit factor
        gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
        gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 0.0
        metrics["profit_factor"] = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        # Best/worst trade
        metrics["best_trade_pct"] = float(np.max(pnl_pcts) * 100)
        metrics["worst_trade_pct"] = float(np.min(pnl_pcts) * 100)

        # Costs
        metrics["total_fees"] = sum(t.fee_cost for t in trades)
        metrics["total_funding_cost"] = sum(t.funding_cost for t in trades)

        # Expectancy (average expected PnL per trade)
        win_rate_frac = len(wins) / n_trades
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss_abs = abs(np.mean(losses)) if len(losses) > 0 else 0
        metrics["expectancy"] = win_rate_frac * avg_win - (1 - win_rate_frac) * avg_loss_abs
    else:
        metrics["win_rate"] = 0.0
        metrics["avg_trade_pnl"] = 0.0
        metrics["avg_trade_pnl_pct"] = 0.0
        metrics["avg_win"] = 0.0
        metrics["avg_loss"] = 0.0
        metrics["profit_factor"] = 0.0
        metrics["best_trade_pct"] = 0.0
        metrics["worst_trade_pct"] = 0.0
        metrics["total_fees"] = 0.0
        metrics["total_funding_cost"] = 0.0
        metrics["expectancy"] = 0.0

    # --- Exposure ---
    non_zero_bars = np.sum(returns != 0)
    metrics["exposure_pct"] = non_zero_bars / n * 100 if n > 0 else 0.0

    # --- Backtest meta ---
    metrics["num_bars"] = n
    metrics["backtest_duration_s"] = result.backtest_duration_s

    return metrics


def print_metrics(metrics: dict, title: str = "Performance Metrics"):
    """Print formatted metrics table."""
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")

    sections = [
        ("Returns", [
            ("Total Return", f"{metrics['total_return_pct']:+.2f}%"),
            ("Annual Return (CAGR)", f"{metrics['annual_return_pct']:+.2f}%"),
            ("Volatility (Ann.)", f"{metrics['volatility_annual_pct']:.2f}%"),
        ]),
        ("Risk-Adjusted", [
            ("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}"),
            ("Sortino Ratio", f"{metrics['sortino_ratio']:.3f}"),
            ("Calmar Ratio", f"{metrics['calmar_ratio']:.3f}"),
        ]),
        ("Drawdown", [
            ("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%"),
            ("Max DD Duration", f"{metrics['max_drawdown_duration_bars']} bars"),
        ]),
        ("Trades", [
            ("Total Trades", f"{metrics['num_trades']}"),
            ("Win Rate", f"{metrics['win_rate']:.1f}%"),
            ("Profit Factor", f"{metrics['profit_factor']:.2f}"),
            ("Avg Trade PnL", f"${metrics['avg_trade_pnl']:.2f}"),
            ("Avg Win", f"${metrics['avg_win']:.2f}"),
            ("Avg Loss", f"${metrics['avg_loss']:.2f}"),
            ("Expectancy", f"${metrics['expectancy']:.2f}"),
        ]),
        ("Costs", [
            ("Total Fees", f"${metrics['total_fees']:.2f}"),
            ("Total Funding", f"${metrics['total_funding_cost']:.2f}"),
        ]),
        ("Other", [
            ("Market Exposure", f"{metrics['exposure_pct']:.1f}%"),
            ("Skewness", f"{metrics['skewness']:.3f}"),
            ("Kurtosis", f"{metrics['kurtosis']:.3f}"),
            ("Backtest Time", f"{metrics['backtest_duration_s']:.1f}s"),
        ]),
    ]

    for section_name, items in sections:
        print(f"\n  {section_name}:")
        for label, value in items:
            print(f"    {label:<25s} {value:>15s}")

    print(f"\n{'=' * 55}")


def metrics_to_tsv_row(
    metrics: dict,
    strategy_name: str,
    symbol: str,
    commit: str = "",
    status: str = "keep",
    description: str = "",
    period: str = "train",
) -> str:
    """Format metrics as a TSV row for results.tsv."""
    fields = [
        commit,
        strategy_name,
        symbol,
        f"{metrics['sharpe_ratio']:.4f}",
        f"{metrics['total_return_pct']:.2f}",
        f"{metrics['annual_return_pct']:.2f}",
        f"{metrics['max_drawdown_pct']:.2f}",
        f"{metrics['win_rate']:.1f}",
        f"{metrics['profit_factor']:.2f}",
        f"{metrics['num_trades']}",
        f"{metrics['sortino_ratio']:.4f}",
        f"{metrics['calmar_ratio']:.4f}",
        status,
        description,
        period,
    ]
    return "\t".join(fields)


TSV_HEADER = "\t".join([
    "commit", "strategy", "symbol", "sharpe", "return_pct", "cagr_pct",
    "max_dd_pct", "win_rate", "profit_factor", "trades",
    "sortino", "calmar", "status", "description", "period"
])
