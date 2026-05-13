#!/usr/bin/env python3
# 6h_WeeklyPivot_Trend_1dVolume
# Hypothesis: Use 1w pivot points for directional bias (above/below pivot) and 1d volume surge for entry.
# Long when price > weekly pivot and daily volume > 2x 20-day average; short when price < weekly pivot and volume surge.
# Exit when price crosses the weekly pivot. Weekly pivot provides structural support/resistance in both bull/bear markets.

name = "6h_WeeklyPivot_Trend_1dVolume"
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
    # Calculate weekly pivot: (H + L + C) / 3
    weekly_pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    pivot_values = weekly_pivot.values
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_values)

    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    vol_avg_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if np.isnan(pivot_aligned[i]) or np.isnan(vol_avg_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price above weekly pivot + volume surge
            if (close[i] > pivot_aligned[i] and
                volume[i] > vol_avg_aligned[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: price below weekly pivot + volume surge
            elif (close[i] < pivot_aligned[i] and
                  volume[i] > vol_avg_aligned[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below weekly pivot
            if close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above weekly pivot
            if close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals