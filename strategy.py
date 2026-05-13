#!/usr/bin/env python3
# 4h_Donchian20_Breakout_12hEMA50_Trend_Volume
# Hypothesis: Use Donchian channel breakout with 12h EMA50 trend filter and volume confirmation for 4h timeframe.
# Long when price breaks above Donchian upper band in uptrend with volume spike.
# Short when price breaks below Donchian lower band in downtrend with volume spike.
# Exit when price returns to Donchian middle band or trend changes.
# Donchian channels provide robust breakout signals, EMA50 filters trend direction, volume confirms strength.
# Designed for moderate trade frequency (75-200 total trades over 4 years) with clear entry/exit rules.

name = "4h_Donchian20_Breakout_12hEMA50_Trend_Volume"
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

    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)

    # Calculate Donchian channel (20-period) on 4h timeframe
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_max_20 + low_min_20) / 2

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian upper band + price above 12h EMA50 (uptrend) + volume spike
            if (close[i] > high_max_20[i] and 
                close[i] > ema_50_12h_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower band + price below 12h EMA50 (downtrend) + volume spike
            elif (close[i] < low_min_20[i] and 
                  close[i] < ema_50_12h_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to Donchian middle band or trend changes (price below EMA50)
            if (close[i] <= mid_20[i] or close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to Donchian middle band or trend changes (price above EMA50)
            if (close[i] >= mid_20[i] or close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals