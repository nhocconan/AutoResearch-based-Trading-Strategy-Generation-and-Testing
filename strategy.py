#!/usr/bin/env python3
# 4h_Donchian20_Breakout_1dTrend_Volume
# Hypothesis: Use 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above upper band in uptrend with volume spike.
# Short when price breaks below lower band in downtrend with volume spike.
# Exit when price returns to the Donchian middle band (20-period SMA).
# Designed for 20-50 trades/year per symbol with clear entry/exit rules to avoid overtrading.
# Works in both bull and bear markets due to trend filter and volume confirmation.

name = "4h_Donchian20_Breakout_1dTrend_Volume"
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

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Calculate Donchian channels (20-period) on 4h data
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle = (high_max + low_min) / 2

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above upper Donchian band + price above 1d EMA50 (uptrend) + volume spike
            if (close[i] > high_max[i] and 
                close[i] > ema_50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian band + price below 1d EMA50 (downtrend) + volume spike
            elif (close[i] < low_min[i] and 
                  close[i] < ema_50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to Donchian middle band
            if close[i] <= middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to Donchian middle band
            if close[i] >= middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals