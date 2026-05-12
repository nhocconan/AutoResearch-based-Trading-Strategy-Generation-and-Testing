#!/usr/bin/env python3
"""
1d_EquityCurveMomentum
Hypothesis: Daily strategy using equity curve momentum (equity curve slope > 0) combined with weekly trend filter and volume confirmation. Works in both bull and bear markets by only taking long positions when the equity curve is trending upward and weekly trend is up, avoiding short exposure in prolonged bear markets while capturing long-side momentum. Target: 15-25 trades/year.
"""

name = "1d_EquityCurveMomentum"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter (call once before loop)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    close_weekly = df_weekly['close'].values
    sma50_weekly = pd.Series(close_weekly).rolling(window=50, min_periods=50).mean().values
    sma50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, sma50_weekly)

    # Equity curve momentum: 20-day slope of equity curve (assuming 100% long)
    # Equity curve = cumulative returns from 100% long position
    returns = np.diff(close, prepend=close[0]) / close
    equity_curve = np.cumprod(1 + returns)
    equity_slope = pd.Series(equity_curve).diff(20) / 20  # 20-period slope
    equity_slope = np.concatenate([np.full(20, np.nan), equity_slope[20:]])  # align length

    # Volume confirmation: volume > 1.5x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long

    for i in range(100, n):
        eq_slope = equity_slope[i]
        sma50_weekly_val = sma50_weekly_aligned[i]
        vol_avg_val = vol_avg_20[i]

        if np.isnan(eq_slope) or np.isnan(sma50_weekly_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: equity curve slope > 0 + price above weekly SMA50 + volume confirmation
            if eq_slope > 0 and close[i] > sma50_weekly_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT: equity curve slope <= 0 or price below weekly SMA50
            if eq_slope <= 0 or close[i] < sma50_weekly_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25

    return signals