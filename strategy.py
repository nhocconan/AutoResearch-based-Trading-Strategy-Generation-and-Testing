#!/usr/bin/env python3
"""
6h_FisherTransform_1dTrend
Hypothesis: Ehlers Fisher Transform identifies turning points with leading signals. 
Applied on 6h price, filtered by 1d EMA trend (EMA34) to trade in direction of higher timeframe trend.
In bull market: long when Fisher crosses above -1.5 with 1d uptrend. 
In bear market: short when Fisher crosses below +1.5 with 1d downtrend.
Uses volume confirmation to filter weak signals. Targets 15-35 trades/year.
"""

name = "6h_FisherTransform_1dTrend"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Ehlers Fisher Transform (9 period)
    price = (high + low) / 2
    max_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    min_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    range_val = max_high - min_low
    # Avoid division by zero
    value1 = np.where(range_val != 0, 2 * ((price - min_low) / range_val - 0.5), 0)
    # Smooth value1
    value2 = np.zeros_like(value1)
    for i in range(1, n):
        value2[i] = 0.33 * value1[i] + 0.67 * value2[i-1]
    # Clamp to [-0.99, 0.99] to avoid domain errors in log
    value2 = np.clip(value2, -0.99, 0.99)
    fish = 0.5 * np.log((1 + value2) / (1 - value2)) + 0.5 * np.roll(fish if 'fish' in locals() else np.zeros(n), 1)
    # Initialize first value
    if n > 0:
        fish[0] = 0
    for i in range(1, n):
        fish[i] = 0.5 * np.log((1 + value2[i]) / (1 - value2[i])) + 0.5 * fish[i-1]

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(10, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_20[i]) or np.isnan(fish[i]) or np.isnan(fish[i-1]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Fisher crosses above -1.5 + 1d uptrend + volume spike
            if fish[i-1] <= -1.5 and fish[i] > -1.5 and close[i] > ema34_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Fisher crosses below +1.5 + 1d downtrend + volume spike
            elif fish[i-1] >= 1.5 and fish[i] < 1.5 and close[i] < ema34_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Fisher crosses below +1.5 or 1d trend turns down
            if fish[i-1] >= 1.5 and fish[i] < 1.5 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Fisher crosses above -1.5 or 1d trend turns up
            if fish[i-1] <= -1.5 and fish[i] > -1.5 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals