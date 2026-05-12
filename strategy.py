#!/usr/bin/env python3
"""
6h_WeeklyPivot_Breakout_WeeklyTrend
Hypothesis: On 6h timeframe, buy when price breaks above the weekly pivot point with price above weekly 200 EMA (bullish trend) and volume > 1.5x average; sell when price breaks below weekly pivot with price below weekly 200 EMA (bearish trend) and volume > 1.5x average. Uses weekly timeframe for trend and pivot levels to capture major market structure, reducing false breakouts in both bull and bear markets. Targets 15-30 trades per year to minimize fee drag.
"""

name = "6h_WeeklyPivot_Breakout_WeeklyTrend"
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

    # Get weekly data for pivot points and trend filter
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values

    # Calculate weekly pivot point (standard formula)
    # Pivot = (High + Low + Close) / 3
    pivot_w = (high_w + low_w + close_w) / 3

    # Use previous weekly bar's pivot (shift by 1)
    pivot_w_prev = np.roll(pivot_w, 1)
    pivot_w_prev[0] = np.nan

    # Align weekly pivot to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w_prev)

    # Weekly EMA200 for trend filter
    ema200_w = pd.Series(close_w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_w_aligned = align_htf_to_ltf(prices, df_w, ema200_w)

    # Volume confirmation: volume > 1.5x 50-period average (approx 12.5 days on 6h)
    vol_avg_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(pivot_w_aligned[i]) or np.isnan(ema200_w_aligned[i]) or 
            np.isnan(vol_avg_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above weekly pivot + weekly uptrend + volume spike
            if (close[i] > pivot_w_aligned[i] and 
                close[i] > ema200_w_aligned[i] and 
                volume[i] > vol_avg_50[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly pivot + weekly downtrend + volume spike
            elif (close[i] < pivot_w_aligned[i] and 
                  close[i] < ema200_w_aligned[i] and 
                  volume[i] > vol_avg_50[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below weekly pivot OR trend turns down
            if close[i] < pivot_w_aligned[i] or close[i] < ema200_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above weekly pivot OR trend turns up
            if close[i] > pivot_w_aligned[i] or close[i] > ema200_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals