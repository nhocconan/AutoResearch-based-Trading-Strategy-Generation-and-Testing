#!/usr/bin/env python3
# 4h_Donchian20_Breakout_1dTrend_Volume_v2
# Hypothesis: Use Donchian(20) breakout on 4h with 1d EMA200 trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel in uptrend with volume spike, short when breaks below lower in downtrend with volume spike.
# Exit when price returns to Donchian middle (mean of 20-period high/low) or trend changes.
# Designed for low-to-moderate trade frequency (~150 total trades over 4 years) with clear rules to avoid overtrading and work in both bull and bear markets.

name = "4h_Donchian20_Breakout_1dTrend_Volume_v2"
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

    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200 for trend filter
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)

    # Calculate Donchian channels on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_donch = high_series.rolling(window=20, min_periods=20).max().values
    lower_donch = low_series.rolling(window=20, min_periods=20).min().values
    middle_donch = (upper_donch + lower_donch) / 2.0

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(upper_donch[i]) or np.isnan(lower_donch[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above upper Donchian + price above 1d EMA200 (uptrend) + volume spike
            if (close[i] > upper_donch[i] and 
                close[i] > ema_200_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian + price below 1d EMA200 (downtrend) + volume spike
            elif (close[i] < lower_donch[i] and 
                  close[i] < ema_200_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to middle Donchian or trend changes (price below EMA200)
            if (close[i] <= middle_donch[i] or close[i] < ema_200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to middle Donchian or trend changes (price above EMA200)
            if (close[i] >= middle_donch[i] or close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals