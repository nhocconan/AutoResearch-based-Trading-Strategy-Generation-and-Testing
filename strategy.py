#!/usr/bin/env python3
"""
6h_FisherTransform_1dTrend_VolumeFilter
Hypothesis: Ehlers Fisher Transform (length=8) on 6h provides early reversal signals. 
Filtered by 1d EMA50 trend direction and volume > 1.3x 20-period average. 
Trades only in direction of 1d trend to avoid counter-trend whipsaws. 
Designed for low trade frequency (~20-40/year) to minimize fee drag on 6h timeframe.
"""

name = "6h_FisherTransform_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def fisher_transform(hlcc4, length=8):
    """Ehlers Fisher Transform. Returns values in [-inf, +inf]."""
    # Normalize price to [-1, 1] range
    max_h = np.max(hlcc4)
    min_l = np.min(hlcc4)
    if max_h == min_l:
        return np.zeros_like(hlcc4)
    value1 = 2 * ((hlcc4 - min_l) / (max_h - min_l) - 0.5)
    # Smooth
    value1 = np.where(np.isnan(value1), 0, value1)
    value2 = np.copy(value1)
    for i in range(1, len(value1)):
        value2[i] = 0.33 * value1[i] + 0.67 * value2[i-1]
    # Fisher transform
    value2 = np.clip(value2, -0.999, 0.999)
    fish = 0.5 * np.log((1 + value2) / (1 - value2)) * np.sqrt(2)
    return fish

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Calculate HLCC4 for Fisher Transform: (High + Low + 2*Close)/4
    hlcc4 = (high + low + 2 * close) / 4

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Fisher Transform on 6h
    fish = fisher_transform(hlcc4, length=8)

    # Volume filter: >1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 warmup
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(fish[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Fisher crosses above -1.5 + 1d uptrend + volume filter
            if (fish[i] > -1.5 and fish[i-1] <= -1.5 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Fisher crosses below +1.5 + 1d downtrend + volume filter
            elif (fish[i] < 1.5 and fish[i-1] >= 1.5 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Fisher crosses below 0 (exit long)
            if fish[i] < 0 and fish[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Fisher crosses above 0 (exit short)
            if fish[i] > 0 and fish[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals