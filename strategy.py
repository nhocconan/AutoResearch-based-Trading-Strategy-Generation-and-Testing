#!/usr/bin/env python3
# 6h_TRIX_ZeroCross_VolumeFilter
# Hypothesis: TRIX (triple EMA) zero cross with volume confirmation acts as a momentum filter.
# In trending markets, TRIX zero crosses signal acceleration; in ranging markets, volume filter reduces whipsaws.
# Works in both bull and bear by capturing momentum shifts. Uses 1d trend filter to avoid counter-trend trades.
# Target: 50-150 total trades over 4 years with discrete position sizing to minimize fee drag.

name = "6h_TRIX_ZeroCross_VolumeFilter"
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

    # Get 1d data for TRIX calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate TRIX: triple EMA of closing prices
    # TRIX = EMA(EMA(EMA(close, period), period), period)
    close_series = pd.Series(df_1d['close'])
    ema1 = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ((ema3 / ema3.shift(1)) - 1) * 100  # Percentage rate of change
    trix = trix.fillna(0).values  # Handle initial NaN from shift
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 6h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Volume filter: >1.3x 20-period average (more sensitive than 1.5x for 6h)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(trix_aligned[i-1]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + price above 1d EMA50 (uptrend) + volume spike
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and 
                close[i] > ema_50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.3):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + price below 1d EMA50 (downtrend) + volume spike
            elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and 
                  close[i] < ema_50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.3):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero or trend changes (price below EMA50)
            if (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0) or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero or trend changes (price above EMA50)
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0) or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals