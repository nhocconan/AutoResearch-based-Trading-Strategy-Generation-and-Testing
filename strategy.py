#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Reversal_Confirmation
Hypothesis: Weekly pivot points (from prior week) act as strong support/resistance in 6h timeframe.
Price rejecting at weekly R1/S1 with confirmation from 1d trend and volume spike indicates reversal.
Works in bull/bear markets: weekly pivots adapt to volatility, providing dynamic levels.
Target: 15-25 trades/year per symbol with strict entry conditions to minimize fee drag.
"""

name = "6h_Weekly_Pivot_Reversal_Confirmation"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_pivot_points(high, low, close):
    """Calculate standard pivot points: P, R1, R2, S1, S2"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, r2, s1, s2

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for pivot points (prior week's data)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 5:
        return np.zeros(n)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values

    # Calculate pivot points from previous week
    _, weekly_r1, _, weekly_s1, _ = calculate_pivot_points(
        weekly_high, weekly_low, weekly_close
    )

    # Align weekly pivot levels to 6h timeframe (using prior week's values)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    # 1d EMA20 for trend
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or
            np.isnan(ema20_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price touches/slightly breaks S1 then reverses up + 1d uptrend + volume spike
            # Condition: low touches S1 and close rebounds above S1
            touches_support = low[i] <= weekly_s1_aligned[i] * 1.001  # allow 0.1% slack
            closes_above_support = close[i] > weekly_s1_aligned[i]
            if touches_support and closes_above_support and close[i] > ema20_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches/slightly breaks R1 then reverses down + 1d downtrend + volume spike
            # Condition: high touches R1 and close falls below R1
            touches_resistance = high[i] >= weekly_r1_aligned[i] * 0.999  # allow 0.1% slack
            closes_below_resistance = close[i] < weekly_r1_aligned[i]
            if touches_resistance and closes_below_resistance and close[i] < ema20_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches R1 or 1d trend turns down
            if close[i] >= weekly_r1_aligned[i] or close[i] < ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches S1 or 1d trend turns up
            if close[i] <= weekly_s1_aligned[i] or close[i] > ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals