#!/usr/bin/env python3
# 12h_Donchian20_Breakout_1dTrend_Volume
# Hypothesis: Use 12h Donchian(20) breakouts with 1d EMA100 trend filter and volume confirmation (>1.5x 20-period average). Designed for low trade frequency (20-50 total trades over 4 years) to avoid fee drag. Works in bull via breakouts, bear via short breakdowns.

name = "12h_Donchian20_Breakout_1dTrend_Volume"
timeframe = "12h"
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

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA100 for trend filter
    ema_100_1d = pd.Series(df_1d['close']).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    # Calculate 12h Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(ema_100_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian high + price above 1d EMA100 (uptrend) + volume spike
            if (close[i] > high_max_20[i] and 
                close[i] > ema_100_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + price below 1d EMA100 (downtrend) + volume spike
            elif (close[i] < low_min_20[i] and 
                  close[i] < ema_100_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to Donchian low or trend changes (price below EMA100)
            if (close[i] <= low_min_20[i] or close[i] < ema_100_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to Donchian high or trend changes (price above EMA100)
            if (close[i] >= high_max_20[i] or close[i] > ema_100_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals