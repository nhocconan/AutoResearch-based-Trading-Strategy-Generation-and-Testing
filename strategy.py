#!/usr/bin/env python3

# 12h_WeeklyPivot_R1_S1_Breakout_TrendVolume
# Hypothesis: Price reactions at weekly Camarilla pivot levels (R1, S1) provide high-probability breakout/reversal signals.
# Uses 1d trend filter (price above/below 200 EMA) to align with higher timeframe momentum.
# Volume spike confirms institutional participation. Designed for low-frequency, high-quality setups to minimize fee drag.
# Works in both bull and bear markets by following the 1d trend direction.

name = "12h_WeeklyPivot_R1_S1_Breakout_TrendVolume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate weekly pivot points and R1/S1 levels
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w

    # Align weekly pivot levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)

    # 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)

    # Volume spike: volume > 2.0 * 24-period average (~12 days at 12h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * vol_ma_24

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema200_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Uptrend + price breaks above R1 + volume spike
            if close[i] > ema200_aligned[i] and close[i] > r1_aligned[i] * 1.001 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend + price breaks below S1 + volume spike
            elif close[i] < ema200_aligned[i] and close[i] < s1_aligned[i] * 0.999 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below pivot or trend turns bearish
            if close[i] < pivot_aligned[i] * 0.999 or close[i] < ema200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above pivot or trend turns bullish
            if close[i] > pivot_aligned[i] * 1.001 or close[i] > ema200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals