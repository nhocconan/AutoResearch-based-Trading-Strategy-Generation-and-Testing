#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeFilter
Hypothesis: On 4h timeframe, price breaking above/below Donchian channel (20-period high/low) indicates momentum continuation. Use 12h EMA50 as trend filter (price > EMA50 for long, < EMA50 for short) and volume > 2x 20-period average for confirmation. Exit on opposite Donchian break or trend reversal. Targets 20-40 trades/year to avoid fee drag.
"""

name = "4h_Donchian20_Breakout_12hTrend_VolumeFilter"
timeframe = "4h"
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

    # Get 12h data (call once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)

    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values

    # Calculate 12h EMA50 for trend
    ema50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values

    # Calculate 4h Donchian channel (20-period high/low)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Volume confirmation: 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # track holding period

    for i in range(50, n):
        # Get aligned values for current 4h bar
        ema50_a = align_htf_to_ltf(prices, df_12h, ema50)[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(ema50_a) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian high + price above EMA50 + volume surge
            if (close[i] > high_max[i] and 
                close[i] > ema50_a and 
                volume[i] > vol_avg_val * 2.0):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # SHORT: Price breaks below Donchian low + price below EMA50 + volume surge
            elif (close[i] < low_min[i] and 
                  close[i] < ema50_a and 
                  volume[i] > vol_avg_val * 2.0):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            bars_since_entry += 1
            # EXIT LONG: Price breaks below Donchian low or price below EMA50
            if (close[i] < low_min[i] or close[i] < ema50_a):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            bars_since_entry += 1
            # EXIT SHORT: Price breaks above Donchian high or price above EMA50
            if (close[i] > high_max[i] or close[i] > ema50_a):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25

    return signals