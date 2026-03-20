#!/usr/bin/env python3
"""
backtest.py - Vectorized Backtesting Engine
============================================
IMMUTABLE FILE - Do not modify during research experiments.

Honest simulation rules:
1. Signal at bar t -> fill at bar t+1 open price
2. Taker fee: 0.04% per side
3. Slippage: 0.01% per side
4. Funding rate applied every 8h to open positions
5. No look-ahead bias (enforced by shifted signal array)
6. Configurable leverage (default 1x, max 20x)

Usage:
    python backtest.py                          # Run current strategy.py on train data
    python backtest.py --test                   # Run on test data
    python backtest.py --symbol ETHUSDT         # Specific symbol
    python backtest.py --timeframe 4h           # Specific timeframe
    python backtest.py --all-symbols            # Run on all symbols
"""

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from prepare import load_config, load_klines, load_funding_rate, get_train_test_split


# =============================================================================
# Data Types
# =============================================================================

@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    taker_fee_pct: float = 0.04
    slippage_pct: float = 0.01
    fill_delay_bars: int = 1
    default_leverage: float = 1.0
    max_leverage: float = 20.0
    initial_capital: float = 10000.0
    include_funding: bool = True

    @classmethod
    def from_config(cls, config: dict) -> "BacktestConfig":
        bt = config["backtest"]
        return cls(
            taker_fee_pct=bt["taker_fee_pct"],
            slippage_pct=bt["slippage_pct"],
            fill_delay_bars=bt["fill_delay_bars"],
            default_leverage=bt["default_leverage"],
            max_leverage=bt["max_leverage"],
            initial_capital=bt["initial_capital"],
            include_funding=bt["include_funding"],
        )


@dataclass
class TradeRecord:
    """Single trade record."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int  # 1 = long, -1 = short
    entry_price: float
    exit_price: float
    size: float  # position size in base currency
    leverage: float
    pnl: float  # realized PnL after costs
    pnl_pct: float  # PnL as percentage of capital used
    funding_cost: float
    fee_cost: float


@dataclass
class BacktestResult:
    """Complete backtest results."""
    # Identifiers
    symbol: str
    timeframe: str
    strategy_name: str
    period: str  # "train" or "test"

    # Equity curve
    equity_curve: np.ndarray  # equity value at each bar
    returns: np.ndarray  # per-bar returns

    # Trade list
    trades: list[TradeRecord] = field(default_factory=list)

    # Summary metrics (computed by evaluate.py)
    metrics: dict = field(default_factory=dict)

    # Timing
    backtest_duration_s: float = 0.0
    num_bars: int = 0


# =============================================================================
# Core Backtesting Engine
# =============================================================================

def _build_funding_per_bar(
    open_times: np.ndarray,
    funding_df: Optional[pd.DataFrame],
) -> np.ndarray:
    """Pre-compute funding rate for each bar using searchsorted (O(n+m))."""
    n = len(open_times)
    funding_per_bar = np.zeros(n, dtype=np.float64)

    if funding_df is None or len(funding_df) == 0:
        return funding_per_bar

    # Convert funding times to numpy datetime64 for fast comparison
    funding_times = funding_df["calc_time"].values.astype("datetime64[ns]")
    funding_rates = funding_df["last_funding_rate"].values.astype(np.float64)

    # For each funding event, find which bar it falls into
    # searchsorted gives us the index where funding_time would be inserted
    bar_times = open_times.astype("datetime64[ns]")
    indices = np.searchsorted(bar_times, funding_times, side="right") - 1

    # Assign funding rates to their respective bars
    for idx, rate in zip(indices, funding_rates):
        if 0 <= idx < n:
            funding_per_bar[idx] += rate

    return funding_per_bar


def run_backtest(
    signals: np.ndarray,
    prices: pd.DataFrame,
    funding_df: Optional[pd.DataFrame],
    bt_config: BacktestConfig,
    leverage: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, list[TradeRecord]]:
    """
    Run backtest with vectorized pre-computation and efficient loop.

    Args:
        signals: Array of position signals. Values:
            +1.0 = full long, -1.0 = full short, 0.0 = flat
            Fractional values allowed (e.g., 0.5 = half position)
        prices: DataFrame with columns: open_time, open, high, low, close, volume
        funding_df: DataFrame with columns: calc_time, last_funding_rate
        bt_config: Backtesting configuration
        leverage: Position leverage multiplier

    Returns:
        (equity_curve, returns, trades)
    """
    n = len(prices)
    assert len(signals) == n, f"Signal length {len(signals)} != price length {n}"

    # Clamp leverage
    leverage = min(leverage, bt_config.max_leverage)

    # Cost per trade (one side)
    cost_pct = (bt_config.taker_fee_pct + bt_config.slippage_pct) / 100.0

    # Extract price arrays
    open_prices = prices["open"].values.astype(np.float64)
    close_prices = prices["close"].values.astype(np.float64)
    open_times = prices["open_time"].values  # numpy datetime64

    # Shift signals by fill_delay_bars to enforce no look-ahead
    delayed_signals = np.zeros(n, dtype=np.float64)
    delay = bt_config.fill_delay_bars
    if delay > 0 and delay < n:
        delayed_signals[delay:] = signals[:-delay]

    # Pre-compute funding rates per bar (O(n+m) instead of O(n*m))
    funding_per_bar = np.zeros(n, dtype=np.float64)
    if bt_config.include_funding:
        funding_per_bar = _build_funding_per_bar(open_times, funding_df)

    # Simulation
    equity = np.zeros(n, dtype=np.float64)
    equity[0] = bt_config.initial_capital
    returns_arr = np.zeros(n, dtype=np.float64)
    trades: list[TradeRecord] = []

    current_position = 0.0
    entry_price = 0.0
    entry_time = None
    cumulative_funding = 0.0

    for i in range(1, n):
        target_position = delayed_signals[i]
        bar_open = open_prices[i]
        bar_close = close_prices[i]
        prev_equity = equity[i - 1]

        # Position change at bar open
        position_change = target_position - current_position
        trade_cost = abs(position_change) * cost_pct * leverage

        # If closing or reversing a position, record the trade
        if current_position != 0.0 and position_change != 0.0:
            close_fraction = min(abs(position_change), abs(current_position))
            if np.sign(position_change) != np.sign(current_position) or target_position == 0.0:
                price_return = (bar_open - entry_price) / entry_price * np.sign(current_position)
                pnl_pct = price_return * leverage * (close_fraction / abs(current_position))
                pnl = prev_equity * pnl_pct * abs(current_position)
                fee_cost = abs(close_fraction) * cost_pct * leverage * prev_equity

                trades.append(TradeRecord(
                    entry_time=entry_time,
                    exit_time=pd.Timestamp(open_times[i]),
                    direction=int(np.sign(current_position)),
                    entry_price=entry_price,
                    exit_price=bar_open,
                    size=abs(current_position),
                    leverage=leverage,
                    pnl=pnl - fee_cost - cumulative_funding,
                    pnl_pct=pnl_pct - trade_cost,
                    funding_cost=cumulative_funding,
                    fee_cost=fee_cost,
                ))
                cumulative_funding = 0.0

        # Update position
        if target_position != 0.0 and current_position == 0.0:
            entry_price = bar_open
            entry_time = pd.Timestamp(open_times[i])
            cumulative_funding = 0.0
        elif target_position != 0.0 and np.sign(target_position) != np.sign(current_position):
            entry_price = bar_open
            entry_time = pd.Timestamp(open_times[i])
            cumulative_funding = 0.0

        current_position = target_position

        # Mark-to-market PnL for the bar (open to close)
        bar_return = (bar_close - bar_open) / bar_open * current_position * leverage if current_position != 0.0 else 0.0

        # Funding cost (pre-computed per bar)
        funding_cost = 0.0
        if current_position != 0.0 and funding_per_bar[i] != 0.0:
            funding_cost = abs(current_position) * funding_per_bar[i] * leverage
            cumulative_funding += abs(prev_equity * funding_cost)

        # Update equity
        bar_pnl = bar_return - trade_cost - funding_cost
        returns_arr[i] = bar_pnl
        equity[i] = prev_equity * (1.0 + bar_pnl)

        # Bankruptcy check
        if equity[i] <= 0:
            equity[i:] = 0.0
            returns_arr[i + 1:] = 0.0
            break

    # Close any remaining position at the end
    if current_position != 0.0:
        final_price = close_prices[-1]
        price_return = (final_price - entry_price) / entry_price * np.sign(current_position)
        pnl_pct = price_return * leverage
        pnl = equity[-2] * pnl_pct * abs(current_position) if len(equity) > 1 else 0
        fee_cost = abs(current_position) * cost_pct * leverage * (equity[-2] if len(equity) > 1 else bt_config.initial_capital)

        trades.append(TradeRecord(
            entry_time=entry_time,
            exit_time=pd.Timestamp(open_times[-1]),
            direction=int(np.sign(current_position)),
            entry_price=entry_price,
            exit_price=final_price,
            size=abs(current_position),
            leverage=leverage,
            pnl=pnl - fee_cost - cumulative_funding,
            pnl_pct=pnl_pct - cost_pct * leverage,
            funding_cost=cumulative_funding,
            fee_cost=fee_cost,
        ))

    return equity, returns_arr, trades


# =============================================================================
# Strategy Loading
# =============================================================================

def load_strategy(strategy_path: str = "strategy.py"):
    """
    Dynamically load a strategy module.

    The strategy module must define:
        - name: str - Strategy name
        - leverage: float - Desired leverage (default 1.0)
        - timeframe: str - Primary timeframe (e.g., "1h")
        - generate_signals(prices: pd.DataFrame) -> np.ndarray
            Takes OHLCV DataFrame, returns signal array of same length.
            Signal values: +1 (long), -1 (short), 0 (flat)

    Optional:
        - extra_timeframes: list[str] - Additional timeframes needed
        - generate_signals_multi(data: dict[str, pd.DataFrame]) -> np.ndarray
            For multi-timeframe strategies. data keys are timeframe strings.
    """
    spec = importlib.util.spec_from_file_location("strategy", strategy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load strategy from {strategy_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Validate required attributes
    required = ["name", "generate_signals"]
    for attr in required:
        if not hasattr(module, attr):
            raise AttributeError(f"Strategy missing required attribute: {attr}")

    return module


# =============================================================================
# Main Backtest Runner
# =============================================================================

def run_strategy_backtest(
    strategy_path: str = "strategy.py",
    symbol: str = "BTCUSDT",
    period: str = "train",
    config: Optional[dict] = None,
) -> BacktestResult:
    """
    Run a complete backtest for a strategy on a specific symbol and period.

    Args:
        strategy_path: Path to strategy.py
        symbol: Trading symbol
        period: "train" or "test"
        config: Optional config dict (loads from file if None)

    Returns:
        BacktestResult with equity curve, trades, and timing info
    """
    start_time = time.time()

    if config is None:
        config = load_config()

    bt_config = BacktestConfig.from_config(config)
    dates = get_train_test_split(config)

    if period == "train":
        start_date, end_date = dates["train_start"], dates["train_end"]
    elif period == "test":
        start_date, end_date = dates["test_start"], dates["test_end"]
    else:
        raise ValueError(f"Invalid period: {period}. Use 'train' or 'test'.")

    # Load strategy
    strategy = load_strategy(strategy_path)
    timeframe = getattr(strategy, "timeframe", "1h")
    leverage = getattr(strategy, "leverage", 1.0)
    strategy_name = strategy.name

    # Load price data
    prices = load_klines(symbol, timeframe, start_date, end_date, config)
    if len(prices) == 0:
        raise ValueError(f"No data for {symbol} {timeframe} in {period} period")

    # Load funding data
    funding_df = None
    if bt_config.include_funding:
        try:
            funding_df = load_funding_rate(symbol, start_date, end_date, config)
        except FileNotFoundError:
            pass  # Proceed without funding data

    # Check for multi-timeframe strategy
    if hasattr(strategy, "generate_signals_multi") and hasattr(strategy, "extra_timeframes"):
        data = {timeframe: prices}
        for tf in strategy.extra_timeframes:
            data[tf] = load_klines(symbol, tf, start_date, end_date, config)
        signals = strategy.generate_signals_multi(data)
    else:
        signals = strategy.generate_signals(prices)

    # Validate signals
    assert len(signals) == len(prices), \
        f"Signal length {len(signals)} != data length {len(prices)}"
    assert np.all(np.isfinite(signals)), "Signals contain NaN or Inf"
    assert np.all(np.abs(signals) <= 1.0 + 1e-9), "Signals must be in [-1, 1]"

    # Run backtest
    equity, returns, trades = run_backtest(
        signals=signals,
        prices=prices,
        funding_df=funding_df,
        bt_config=bt_config,
        leverage=leverage,
    )

    duration = time.time() - start_time

    return BacktestResult(
        symbol=symbol,
        timeframe=timeframe,
        strategy_name=strategy_name,
        period=period,
        equity_curve=equity,
        returns=returns,
        trades=trades,
        backtest_duration_s=duration,
        num_bars=len(prices),
    )


def print_result_summary(result: BacktestResult):
    """Print a concise summary of backtest results."""
    eq = result.equity_curve
    initial = eq[0] if eq[0] > 0 else 1.0
    final = eq[-1]
    total_return = (final - initial) / initial * 100

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    drawdown = (eq - peak) / np.where(peak > 0, peak, 1.0)
    max_dd = np.min(drawdown) * 100

    # Trade stats
    n_trades = len(result.trades)
    if n_trades > 0:
        wins = sum(1 for t in result.trades if t.pnl > 0)
        win_rate = wins / n_trades * 100
        avg_pnl = np.mean([t.pnl for t in result.trades])
        total_fees = sum(t.fee_cost for t in result.trades)
        total_funding = sum(t.funding_cost for t in result.trades)
    else:
        win_rate = 0
        avg_pnl = 0
        total_fees = 0
        total_funding = 0

    print(f"\n{'=' * 50}")
    print(f"Strategy: {result.strategy_name}")
    print(f"Symbol: {result.symbol} | TF: {result.timeframe} | Period: {result.period}")
    print(f"Bars: {result.num_bars:,} | Duration: {result.backtest_duration_s:.1f}s")
    print(f"{'=' * 50}")
    print(f"Total Return:  {total_return:+.2f}%")
    print(f"Final Equity:  ${final:,.2f}")
    print(f"Max Drawdown:  {max_dd:.2f}%")
    print(f"Trades:        {n_trades}")
    print(f"Win Rate:      {win_rate:.1f}%")
    print(f"Avg PnL/Trade: ${avg_pnl:.2f}")
    print(f"Total Fees:    ${total_fees:.2f}")
    print(f"Total Funding: ${total_funding:.2f}")
    print(f"{'=' * 50}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run backtest on current strategy")
    parser.add_argument("--strategy", default="strategy.py", help="Path to strategy file")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--timeframe", default=None, help="Override strategy timeframe")
    parser.add_argument("--test", action="store_true", help="Run on test period (2025+)")
    parser.add_argument("--all-symbols", action="store_true", help="Run on all symbols")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    config = load_config()
    period = "test" if args.test else "train"
    symbols = config["data"]["symbols"] if args.all_symbols else [args.symbol]

    results = []
    for symbol in symbols:
        result = run_strategy_backtest(
            strategy_path=args.strategy,
            symbol=symbol,
            period=period,
            config=config,
        )

        if args.json:
            results.append({
                "symbol": result.symbol,
                "timeframe": result.timeframe,
                "strategy": result.strategy_name,
                "period": result.period,
                "total_return_pct": float((result.equity_curve[-1] - result.equity_curve[0]) / result.equity_curve[0] * 100),
                "final_equity": float(result.equity_curve[-1]),
                "max_drawdown_pct": float(np.min((result.equity_curve - np.maximum.accumulate(result.equity_curve)) / np.maximum.accumulate(result.equity_curve)) * 100),
                "num_trades": len(result.trades),
                "win_rate": float(sum(1 for t in result.trades if t.pnl > 0) / max(len(result.trades), 1) * 100),
                "backtest_duration_s": result.backtest_duration_s,
            })
        else:
            print_result_summary(result)

    if args.json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
