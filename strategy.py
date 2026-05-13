#!/usr/bin/env python3
# 1d_PivotPoint_Breakout_1wTrend
# Hypothesis: Use weekly pivot points as long-term support/resistance on daily timeframe.
# Enter long when price breaks above weekly R1 with volume spike and weekly EMA50 uptrend.
# Enter short when price breaks below weekly S1 with volume spike and weekly EMA50 downtrend.
# Exit when price returns to the previous week's close (weekly C level).
# Uses weekly trend filter to avoid counter-trend trades, reducing whipsaw in ranging markets.
# Designed to work in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend).
# Target: 10-20 trades/year per symbol.

name = "1d_PivotPoint_Breakout_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get weekly data for pivot points and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate weekly pivot points for previous week
    # P = (H + L + C) / 3
    # Range = H - L
    # S1 = C - (Range * 1.1 / 12)
    # R1 = C + (Range * 1.1 / 12)
    P = (high_1w + low_1w + close_1w) / 3.0
    rng = high_1w - low_1w

    S1 = close_1w - (rng * 1.1 / 12)
    R1 = close_1w + (rng * 1.1 / 12)

    # Align pivot levels to daily timeframe (use previous week's levels)
    s1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, R1)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    # Get weekly EMA50 for trend filter
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    # Align previous week's close (C level) for exit
    c_prev = np.roll(close_1w, 1)
    c_prev[0] = np.nan
    c_aligned = align_htf_to_ltf(prices, df_1w, c_prev)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_1w_aligned[i]) or
            np.isnan(c_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above R1 with volume spike and weekly EMA uptrend
            if close[i] > r1_aligned[i] and volume_spike[i] and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1 with volume spike and weekly EMA downtrend
            elif close[i] < s1_aligned[i] and volume_spike[i] and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to previous week's close (C level)
            if close[i] <= c_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to previous week's close (C level)
            if close[i] >= c_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals