#!/usr/bin/env python3
# 6h_WeeklyPivot_Momentum_Catch
# Hypothesis: Use weekly pivot levels (from 1w data) as major support/resistance zones.
# Enter long when price breaks above weekly R1 with 1d EMA uptrend and volume confirmation.
# Enter short when price breaks below weekly S1 with 1d EMA downtrend and volume confirmation.
# Exit when price reaches the opposite weekly pivot level (e.g., long exits at weekly S1).
# Weekly pivots are less noisy than daily and capture major turning points.
# Combined with 1d trend filter and volume spike to avoid false breakouts in chop.
# Target: 15-25 trades/year on 6h to minimize fee decay while capturing strong trending moves.

name = "6h_WeeklyPivot_Momentum_Catch"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate weekly pivot points (based on previous week's OHLC)
    range_1w = high_1w - low_1w
    pp = (high_1w + low_1w + close_1w) / 3.0
    r1 = pp + (high_1w - low_1w) * 1.1 / 4.0
    s1 = pp - (high_1w - low_1w) * 1.1 / 4.0
    r2 = pp + (high_1w - low_w) * 1.1 / 2.0
    s2 = pp - (high_1w - low_1w) * 1.1 / 2.0

    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)

    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 2.0x 30-period average (to filter weak moves)
    vol_avg_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required value is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above weekly R1 + price > 1d EMA34 + volume spike
            if (close[i] > r1_aligned[i] and
                close[i] > ema34_1d_aligned[i] and
                volume[i] > vol_avg_30[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below weekly S1 + price < 1d EMA34 + volume spike
            elif (close[i] < s1_aligned[i] and
                  close[i] < ema34_1d_aligned[i] and
                  volume[i] > vol_avg_30[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses below weekly S1 (opposite side)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses above weekly R1 (opposite side)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals