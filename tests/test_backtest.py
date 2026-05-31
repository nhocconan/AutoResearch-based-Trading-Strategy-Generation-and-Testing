#!/usr/bin/env python3
"""Engine tests: the no-lookahead, cost, and PnL guarantees that make the
simulation 'honest'. These lock in the contract described in CLAUDE.md and
docs/backtesting-rules.md so the immutable engine cannot silently drift."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import BacktestConfig, run_backtest
from conftest import make_flat_ohlcv, make_ohlcv, make_trending_ohlcv

CFG = BacktestConfig(
    taker_fee_pct=0.04,
    slippage_pct=0.01,
    fill_delay_bars=1,
    include_funding=True,
    initial_capital=10_000.0,
)
COST_PER_SIDE = (CFG.taker_fee_pct + CFG.slippage_pct) / 100.0  # fraction


def test_signal_is_filled_at_t_plus_one_open():
    """A signal emitted at bar t must take effect at bar t+1 (no same-bar fill)."""
    prices = make_trending_ohlcv(n=50)
    signals = np.zeros(len(prices))
    k = 10
    signals[k:] = 1.0  # go long from bar k onward

    _, _, trades = run_backtest(signals, prices, None, CFG)

    assert len(trades) == 1
    entry = trades[0]
    # fill_delay_bars=1 → position opens at bar k+1's open, not bar k's.
    assert entry.entry_time == pd.Timestamp(prices["open_time"].iloc[k + 1])
    assert entry.entry_price == prices["open"].iloc[k + 1]


def test_round_trip_cost_on_flat_market():
    """On a flat market with no funding, a full long round trip should cost
    exactly the documented round-trip fee and nothing else."""
    prices = make_flat_ohlcv(n=40)
    signals = np.zeros(len(prices))
    signals[5:15] = 1.0  # enter, hold, exit — one full round trip

    eq, _, trades = run_backtest(signals, prices, None, CFG, leverage=1.0)

    final = eq[-1]
    expected = CFG.initial_capital * (1.0 - 2 * COST_PER_SIDE)
    # Allow tiny float tolerance; cost is entry side + exit side.
    assert abs(final - expected) < CFG.initial_capital * 1e-6
    assert len(trades) == 1


def test_flat_market_no_position_no_cost():
    """No signal → no trades, capital untouched."""
    prices = make_flat_ohlcv(n=30)
    signals = np.zeros(len(prices))
    eq, _, trades = run_backtest(signals, prices, None, CFG)
    assert len(trades) == 0
    assert eq[-1] == CFG.initial_capital


def test_long_profits_in_uptrend():
    prices = make_trending_ohlcv(n=120, slope=1.0)
    signals = np.zeros(len(prices))
    signals[5:] = 1.0
    eq, _, _ = run_backtest(signals, prices, None, CFG)
    assert eq[-1] > CFG.initial_capital


def test_short_loses_in_uptrend():
    prices = make_trending_ohlcv(n=120, slope=1.0)
    signals = np.zeros(len(prices))
    signals[5:] = -1.0
    eq, _, _ = run_backtest(signals, prices, None, CFG)
    assert eq[-1] < CFG.initial_capital


def test_short_profits_in_downtrend():
    close = 200.0 - 0.8 * np.arange(120)
    prices = make_ohlcv(n=120, close=close)
    signals = np.zeros(len(prices))
    signals[5:] = -1.0
    eq, _, _ = run_backtest(signals, prices, None, CFG)
    assert eq[-1] > CFG.initial_capital


def test_position_sizing_scales_pnl():
    """Half size should yield roughly half the PnL of full size (same path)."""
    prices = make_trending_ohlcv(n=120, slope=1.0)
    full = np.zeros(len(prices))
    full[5:] = 1.0
    half = np.zeros(len(prices))
    half[5:] = 0.5

    eq_full, _, _ = run_backtest(full, prices, None, CFG)
    eq_half, _, _ = run_backtest(half, prices, None, CFG)

    pnl_full = eq_full[-1] - CFG.initial_capital
    pnl_half = eq_half[-1] - CFG.initial_capital
    assert pnl_full > pnl_half > 0
    # Compounding makes it inexact, but half-size PnL must be in the right ballpark.
    assert 0.4 < pnl_half / pnl_full < 0.6


def test_signal_length_mismatch_raises():
    prices = make_flat_ohlcv(n=20)
    try:
        run_backtest(np.zeros(5), prices, None, CFG)
    except AssertionError:
        return
    raise AssertionError("expected AssertionError on signal/price length mismatch")


def test_no_lookahead_last_bar_signal_has_no_effect():
    """A signal on the very last bar cannot be filled (no t+1), so it must not
    move equity — a direct guard against same-bar (look-ahead) execution."""
    prices = make_trending_ohlcv(n=60, slope=1.0)
    signals = np.zeros(len(prices))
    signals[-1] = 1.0  # only the last bar
    eq, _, trades = run_backtest(signals, prices, None, CFG)
    assert len(trades) == 0
    assert eq[-1] == CFG.initial_capital
