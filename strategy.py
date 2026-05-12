#!/usr/bin/env python3
# 6h_Donchian20_WeeklyPivotDirection_Volume
# Hypothesis: Donchian(20) breakouts on 6h aligned with weekly pivot direction (from 1w) and volume confirmation.
# Weekly pivot direction acts as trend filter (price above/below weekly pivot). Volume confirms breakout strength.
# Works in bull (breakouts with trend) and bear (fades at extremes via pivot reversal logic implicitly via stops).
# Target: 20-40 trades/year (~80-160 over 4 years) to stay within fee limits.
# Long: Close > Donchian High(20) + close > weekly pivot + volume > 1.5x SMA20
# Short: Close < Donchian Low(20) + close < weekly pivot + volume > 1.5x SMA20
# Exit: Close crosses back inside Donchian channel (opposite side for symmetry)

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Donchian20_WeeklyPivotDirection_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for pivot direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Calculate weekly pivot point (standard: (H+L+C)/3)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0

    # Get Donchian channel (20-period) on 6h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values

    # Volume confirmation: 1.5x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    # Align weekly pivot to 6h (wait for weekly close)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Get aligned values
        pivot_aligned = weekly_pivot_aligned[i]
        dh = donchian_high[i]
        dl = donchian_low[i]
        vol_thresh = volume_threshold[i]

        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned) or np.isnan(dh) or np.isnan(dl) or 
            np.isnan(vol_thresh)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above Donchian high + above weekly pivot + volume spike
            if (close[i] > dh and
                close[i] > pivot_aligned and
                volume[i] > vol_thresh):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below Donchian low + below weekly pivot + volume spike
            elif (close[i] < dl and
                  close[i] < pivot_aligned and
                  volume[i] > vol_thresh):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes back inside Donchian channel (below low)
            if close[i] < dl:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes back inside Donchian channel (above high)
            if close[i] > dh:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals