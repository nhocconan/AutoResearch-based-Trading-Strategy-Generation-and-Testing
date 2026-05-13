#!/usr/bin/env python3
# 12h_Donchian20_Breakout_1wTrend_Volume
# Hypothesis: Use 12h Donchian channel (20-period) breakouts with 1-week EMA trend filter and volume confirmation.
# Long when price breaks above upper Donchian in uptrend with volume spike, short when breaks below lower in downtrend.
# Exit when price returns to the middle of the Donchian channel or trend changes.
# Designed for low trade frequency (20-50 total trades over 4 years) to minimize fee drag and work in both bull and bear markets.

name = "12h_Donchian20_Breakout_1wTrend_Volume"
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

    # Get 12h data for Donchian channel calculation
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Donchian channel (20-period)
    high_20 = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2
    
    # Align 12h Donchian levels to 12h timeframe (same timeframe, so alignment is direct)
    # Since we're using 12h data on 12h timeframe, no alignment needed for the bands
    # But we still need to align to match the 12h bar timing
    high_20_aligned = align_htf_to_ltf(prices, df_12h, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_12h, low_20)
    mid_20_aligned = align_htf_to_ltf(prices, df_12h, mid_20)

    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(mid_20_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above upper Donchian + price above 1w EMA50 (uptrend) + volume spike
            if (close[i] > high_20_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian + price below 1w EMA50 (downtrend) + volume spike
            elif (close[i] < low_20_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to middle of Donchian or trend changes (price below EMA50)
            if (close[i] <= mid_20_aligned[i] or close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to middle of Donchian or trend changes (price above EMA50)
            if (close[i] >= mid_20_aligned[i] or close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals