#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Donchian20_WeeklyPivotTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian breakout and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    # Get 1w data for weekly pivot trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w weekly pivot points (classic floor trader pivots)
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    prev_weekly_close = df_1w['close'].shift(1).values
    
    pivot_point = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    r1 = 2 * pivot_point - prev_weekly_low
    s1 = 2 * pivot_point - prev_weekly_high
    
    # Volume filter: current 1d volume > 1.3 * 20-day average
    vol_series = pd.Series(df_1d['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = df_1d['volume'].values > (vol_ma * 1.3)
    
    # Align all to 6h timeframe
    high_20_6h = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_6h = align_htf_to_ltf(prices, df_1d, low_20)
    pivot_point_6h = align_htf_to_ltf(prices, df_1w, pivot_point)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    volume_filter_6h = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 20)  # Need enough data for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(high_20_6h[i]) or np.isnan(low_20_6h[i]) or
            np.isnan(pivot_point_6h[i]) or np.isnan(r1_6h[i]) or
            np.isnan(s1_6h[i]) or np.isnan(volume_filter_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_channel = high_20_6h[i]
        lower_channel = low_20_6h[i]
        pivot = pivot_point_6h[i]
        r1_val = r1_6h[i]
        s1_val = s1_6h[i]
        vol_filter = volume_filter_6h[i]
        
        if position == 0:
            # Enter long: break above upper Donchian with volume and above weekly pivot
            if close[i] > upper_channel and close[i] > pivot and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower Donchian with volume and below weekly pivot
            elif close[i] < lower_channel and close[i] < pivot and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below weekly pivot (mean reversion to pivot)
            if close[i] < pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above weekly pivot (mean reversion to pivot)
            if close[i] > pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals