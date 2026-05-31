#!/usr/bin/env python3
"""Phase 3 correctness guards.

These tests pin down two honest-simulation properties that previously drifted:

1. Funding is SIGNED by position direction — the side that should earn funding
   actually earns it (a short in a positive-funding regime), instead of both
   sides being charged.
2. The configured trading cost matches the documented 0.10% round trip, so the
   engine, config.yaml, and the docs can never silently disagree again.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import BacktestConfig, run_backtest
from prepare import load_config
from conftest import make_flat_ohlcv

NO_COST = BacktestConfig(
    taker_fee_pct=0.0, slippage_pct=0.0, fill_delay_bars=1,
    include_funding=True, initial_capital=10_000.0,
)


def _funding_df(times, rate):
    """One funding event of `rate` placed inside the series."""
    return pd.DataFrame({
        "calc_time": pd.to_datetime([times]),
        "last_funding_rate": [float(rate)],
    })


def test_short_earns_funding_long_pays_when_rate_positive():
    """Positive funding rate: long pays, short receives. On a flat market with
    zero fees, the short must end ABOVE and the long BELOW starting capital."""
    prices = make_flat_ohlcv(n=40, freq="8h")
    n = len(prices)
    # Funding event time inside bar ~10 (well after the t+1 fill at bar 1).
    event_time = pd.Timestamp(prices["open_time"].iloc[10])
    rate = 0.01  # +1%

    long_sig = np.zeros(n)
    long_sig[1:] = 1.0
    short_sig = np.zeros(n)
    short_sig[1:] = -1.0

    fdf = _funding_df(event_time, rate)
    eq_long, _, _ = run_backtest(long_sig, prices, fdf, NO_COST)
    eq_short, _, _ = run_backtest(short_sig, prices, fdf, NO_COST)

    assert eq_long[-1] < NO_COST.initial_capital, "long should PAY positive funding"
    assert eq_short[-1] > NO_COST.initial_capital, "short should RECEIVE positive funding"
    # Magnitudes should mirror (~1% each way) on a flat, fee-free market.
    long_loss = NO_COST.initial_capital - eq_long[-1]
    short_gain = eq_short[-1] - NO_COST.initial_capital
    assert abs(long_loss - short_gain) < NO_COST.initial_capital * 1e-3


def test_funding_not_always_a_cost():
    """Regression guard for the old abs() bug: a short in positive funding must
    NOT lose money to funding."""
    prices = make_flat_ohlcv(n=40, freq="8h")
    n = len(prices)
    short_sig = np.zeros(n)
    short_sig[1:] = -1.0
    fdf = _funding_df(pd.Timestamp(prices["open_time"].iloc[10]), 0.02)
    eq, _, _ = run_backtest(short_sig, prices, fdf, NO_COST)
    assert eq[-1] > NO_COST.initial_capital


def test_configured_round_trip_cost_is_documented_value():
    """config.yaml must yield exactly the documented 0.10% round trip — the
    single guard against engine/config/docs drift."""
    cfg = BacktestConfig.from_config(load_config())
    per_side = cfg.taker_fee_pct + cfg.slippage_pct
    assert abs(per_side - 0.05) < 1e-9, f"per-side cost is {per_side}%, expected 0.05%"
    round_trip = 2 * per_side
    assert abs(round_trip - 0.10) < 1e-9, f"round trip is {round_trip}%, expected 0.10%"


def test_round_trip_cost_applied_on_flat_market_matches_config():
    """End-to-end: a full round trip on a flat market costs exactly the
    configured round-trip fee (no funding, no market move)."""
    cfg = BacktestConfig.from_config(load_config())
    cfg.include_funding = False
    prices = make_flat_ohlcv(n=40)
    sig = np.zeros(len(prices))
    sig[5:15] = 1.0
    eq, _, _ = run_backtest(sig, prices, None, cfg)
    cost_frac = (cfg.taker_fee_pct + cfg.slippage_pct) / 100.0
    expected = cfg.initial_capital * (1 - cost_frac) ** 2
    assert abs(eq[-1] - expected) < cfg.initial_capital * 1e-6
