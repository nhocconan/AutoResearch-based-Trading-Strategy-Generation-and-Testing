# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Donchian20_WeeklyPivot_VolumeFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (based on previous week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for Donchian channel and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Weekly Pivot Points (based on previous week)
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    weekly_range = prev_week_high - prev_week_low
    
    # Weekly support/resistance levels
    weekly_R1 = weekly_pivot + (weekly_range * 1.0)
    weekly_S1 = weekly_pivot - (weekly_range * 1.0)
    
    # Align weekly levels to 6h
    weekly_R1_aligned = align_htf_to_ltf(prices, df_1w, weekly_R1)
    weekly_S1_aligned = align_htf_to_ltf(prices, df_1w, weekly_S1)
    
    # Donchian channel (20-period) from daily data
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume filter: current 6h volume > 1.3 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 20)  # Need enough data for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_R1_aligned[i]) or np.isnan(weekly_S1_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr1 = weekly_R1_aligned[i]
        ws1 = weekly_S1_aligned[i]
        dh = donchian_high_aligned[i]
        dl = donchian_low_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: price breaks above weekly R1 AND above Donchian high with volume
            if close[i] > wr1 and close[i] > dh and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly S1 AND below Donchian low with volume
            elif close[i] < ws1 and close[i] < dl and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low
            if close[i] < dl:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high
            if close[i] > dh:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals