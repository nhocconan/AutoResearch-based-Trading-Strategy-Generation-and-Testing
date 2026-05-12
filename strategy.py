#162419
#!/usr/bin/env python3
"""
6h_OrderBlock_Equilibrium_12hTrend
Hypothesis: In 6h timeframe, price tends to revert to the 12h equilibrium (EMA50) after touching order blocks (liquidity zones) identified as recent swing highs/lows with volume confirmation. Uses 12h EMA50 as dynamic trend filter and swing points from 12h swing high/low detection. Designed to work in both bull (buy dips to EMA in uptrend) and bear (sell rallies to EMA in downtrend) markets by fading extremes toward equilibrium. Targets 20-40 trades per year to minimize fee drag.
"""

name = "6h_OrderBlock_Equilibrium_12hTrend"
timeframe = "6h"
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

    # Get 12h data for trend filter and swing points
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values

    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # 12h Swing High/Low detection (liquidity zones/order blocks)
    # Swing high: high > previous 2 and next 2 highs
    # Swing low: low < previous 2 and next 2 lows
    swing_high = np.zeros_like(high_12h, dtype=bool)
    swing_low = np.zeros_like(low_12h, dtype=bool)
    
    for i in range(2, len(high_12h) - 2):
        if (high_12h[i] > high_12h[i-1] and high_12h[i] > high_12h[i-2] and
            high_12h[i] > high_12h[i+1] and high_12h[i] > high_12h[i+2]):
            swing_high[i] = True
        if (low_12h[i] < low_12h[i-1] and low_12h[i] < low_12h[i-2] and
            low_12h[i] < low_12h[i+1] and low_12h[i] < low_12h[i+2]):
            swing_low[i] = True

    # Create arrays of swing levels (NaN where no swing)
    swing_high_levels = np.full_like(high_12h, np.nan)
    swing_low_levels = np.full_like(low_12h, np.nan)
    swing_high_levels[swing_high] = high_12h[swing_high]
    swing_low_levels[swing_low] = low_12h[swing_low]

    # Forward fill to get most recent swing level
    def ffilt_nan(arr):
        mask = np.isnan(arr)
        out = arr.copy()
        if np.any(mask):
            idx = np.where(~mask, np.arange(len(arr)), 0)
            np.maximum.accumulate(idx, out=idx)
            out = arr[idx]
        return out
    
    recent_swing_high = ffilt_nan(swing_high_levels)
    recent_swing_low = ffilt_nan(swing_low_levels)

    # Align swing levels to 6h timeframe
    swing_high_aligned = align_htf_to_ltf(prices, df_12h, recent_swing_high)
    swing_low_aligned = align_htf_to_ltf(prices, df_12h, recent_swing_low)

    # Volume confirmation: volume > 1.5x 20-period average on 6h
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(swing_high_aligned[i]) or 
            np.isnan(swing_low_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price touches recent swing low (support) AND closes above it AND volume confirmation AND price > 12h EMA50 (uptrend)
            if (low[i] <= swing_low_aligned[i] and 
                close[i] > swing_low_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5 and
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches recent swing high (resistance) AND closes below it AND volume confirmation AND price < 12h EMA50 (downtrend)
            elif (high[i] >= swing_high_aligned[i] and 
                  close[i] < swing_high_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5 and
                  close[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches 12h EMA50 (equilibrium) OR breaks swing low
            if close[i] >= ema50_12h_aligned[i] or low[i] < swing_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches 12h EMA50 (equilibrium) OR breaks swing high
            if close[i] <= ema50_12h_aligned[i] or high[i] > swing_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals