#!/usr/bin/env python3
# 6h_MarketProfile_ValueArea_Breakout_1dTrend_Volume
# Hypothesis: Use Market Profile Value Area High/Low from 1d as breakout levels. 
# Enter long when price breaks above VAH with 1d EMA uptrend and volume spike.
# Enter short when price breaks below VAL with 1d EMA downtrend and volume spike.
# Exit when price closes back into the Value Area (between VAL and VAH).
# This structure-based approach reduces false breakouts and works in both bull/bear via trend filter.
# Target: 15-25 trades/year on 6h to minimize fee drag.

name = "6h_MarketProfile_ValueArea_Breakout_1dTrend_Volume"
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

    # Get 1d data for Market Profile Value Area calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate simple Value Area using 1d range and close bias
    # VAH = close + 0.7 * (high - low)  # biased toward close
    # VAL = close - 0.7 * (high - low)
    # This approximates the 70% value area without TPO calculation
    range_1d = high_1d - low_1d
    vah = close_1d + 0.7 * range_1d
    val = close_1d - 0.7 * range_1d

    # Align Value Area levels to 6h timeframe
    vah_aligned = align_htf_to_ltf(prices, df_1d, vah)
    val_aligned = align_htf_to_ltf(prices, df_1d, val)

    # 1d EMA34 for trend filter (more responsive than EMA50)
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(vah_aligned[i]) or np.isnan(val_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above VAH + price > 1d EMA34 + volume spike
            if (close[i] > vah_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below VAL + price < 1d EMA34 + volume spike
            elif (close[i] < val_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses back below VAH (return to value area)
            if close[i] < vah_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses back above VAL (return to value area)
            if close[i] > val_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals