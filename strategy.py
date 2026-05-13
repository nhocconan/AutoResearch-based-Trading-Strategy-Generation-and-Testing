#!/usr/bin/env python3
# 6h_FisherTransform_1dTrend_Volume
# Hypothesis: Fisher Transform crossing above/below key levels (-1.5 for long, +1.5 for short) with 1d EMA34 trend filter and volume confirmation captures reversals with controlled trade frequency.
# Works in bull markets via reversals from oversold conditions and in bear markets via reversals from overbought conditions.
# Uses 1d EMA34 to filter trend direction (only trade with the trend) and volume spike for confirmation, reducing false signals.
# Target: 12-37 trades per year per symbol to minimize fee drag.

name = "6h_FisherTransform_1dTrend_Volume"
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

    # Fisher Transform (9-period)
    hl2 = (high + low) / 2
    max_hl2 = pd.Series(hl2).rolling(window=9, min_periods=9).max().values
    min_hl2 = pd.Series(hl2).rolling(window=9, min_periods=9).min().values
    value = np.where((max_hl2 - min_hl2) != 0, 2 * ((hl2 - min_hl2) / (max_hl2 - min_hl2) - 0.5), 0)
    value = np.clip(value, -0.999, 0.999)
    fish = np.zeros_like(value)
    fish[0] = 0
    for i in range(1, n):
        fish[i] = 0.5 * np.log((1 + value[i]) / (1 - value[i])) + 0.5 * fish[i-1]

    # Get 1d data for EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume filter: >2.0x 24-period average (4 trading days)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        # Skip if any required value is NaN
        if (np.isnan(fish[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Fisher crosses above -1.5 + 1d EMA34 uptrend + volume spike
            if (fish[i] > -1.5 and fish[i-1] <= -1.5 and 
                close[i] > ema34_1d_aligned[i] and
                volume[i] > vol_avg_24[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Fisher crosses below +1.5 + 1d EMA34 downtrend + volume spike
            elif (fish[i] < 1.5 and fish[i-1] >= 1.5 and 
                  close[i] < ema34_1d_aligned[i] and
                  volume[i] > vol_avg_24[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Fisher crosses below +1.5 or volatility drop
            if fish[i] < 1.5 and fish[i-1] >= 1.5 or volume[i] < vol_avg_24[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Fisher crosses above -1.5 or volatility drop
            if fish[i] > -1.5 and fish[i-1] <= -1.5 or volume[i] < vol_avg_24[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals