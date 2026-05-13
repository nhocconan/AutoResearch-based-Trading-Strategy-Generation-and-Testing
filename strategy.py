#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian_Breakout
Hypothesis: Breakouts above weekly Donchian high with weekly pivot bias (price > pivot) for longs, and breakouts below weekly Donchian low with price < pivot for shorts, with volume confirmation. Uses weekly structure to filter noise and capture strong trends in both bull and bear markets.
"""

name = "6h_WeeklyPivot_Donchian_Breakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels and pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period)
    high_20 = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Weekly pivot point (standard calculation)
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    prev_weekly_close = df_1w['close'].shift(1).values
    
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3
    
    # Align weekly indicators to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        if position == 0:
            # LONG: Break above weekly Donchian high with price above weekly pivot and volume
            if (donchian_high_aligned[i] > 0 and not np.isnan(donchian_high_aligned[i]) and
                high[i] > donchian_high_aligned[i] and
                close[i] > pivot_aligned[i] and
                volume_confirmed[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below weekly Donchian low with price below weekly pivot and volume
            elif (donchian_low_aligned[i] > 0 and not np.isnan(donchian_low_aligned[i]) and
                  low[i] < donchian_low_aligned[i] and
                  close[i] < pivot_aligned[i] and
                  volume_confirmed[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below weekly pivot or Donchian low
            if close[i] < pivot_aligned[i] or low[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above weekly pivot or Donchian high
            if close[i] > pivot_aligned[i] or high[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals