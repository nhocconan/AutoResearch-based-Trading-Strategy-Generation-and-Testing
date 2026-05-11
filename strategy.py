#!/usr/bin/env python3
name = "6h_Donchian_Breakout_WeeklyPivot_Direction_12hVolume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian and volume filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1d data for weekly pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # 12h volume filter: current volume > 1.5x 24-period average (3 days on 12h)
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=24, min_periods=24).mean().values
    volume_filter = vol_12h > (vol_ma_12h * 1.5)
    volume_filter_aligned = align_htf_to_ltf(prices, df_12h, volume_filter)
    
    # Weekly pivot from 1d data (weekly high/low/close)
    # We'll approximate weekly by using the last 5 days of 1d data
    weekly_high = pd.Series(df_1d['high'].values).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(df_1d['low'].values).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(df_1d['close'].values).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_range = weekly_high - weekly_low
    # Weekly resistance/support levels
    weekly_r1 = weekly_pivot + (weekly_range * 1.1 / 12)  # R1
    weekly_s1 = weekly_pivot - (weekly_range * 1.1 / 12)  # S1
    weekly_r2 = weekly_pivot + (weekly_range * 1.1 / 6)   # R2
    weekly_s2 = weekly_pivot - (weekly_range * 1.1 / 6)   # S2
    
    # Align weekly levels to 6h
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or np.isnan(weekly_r2_aligned[i]) or
            np.isnan(weekly_s2_aligned[i]) or np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND above weekly pivot AND volume filter
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > weekly_pivot_aligned[i] and 
                volume_filter_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND below weekly pivot AND volume filter
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < weekly_pivot_aligned[i] and 
                  volume_filter_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below Donchian low OR below weekly S1
            if close[i] < donchian_low_aligned[i] or close[i] < weekly_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above Donchian high OR above weekly R1
            if close[i] > donchian_high_aligned[i] or close[i] > weekly_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals