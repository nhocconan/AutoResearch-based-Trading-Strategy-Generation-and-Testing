#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using Friday's data)
    # We'll use the last available daily data as proxy for weekly
    # For true weekly pivot, we would need: (weekly high + weekly low + weekly close) / 3
    # Since we don't have weekly aggregation, we use daily with lookback
    
    # Calculate 5-day rolling high/low/close for weekly approximation
    if len(high_1d) >= 5:
        weekly_high = np.full(len(high_1d), np.nan)
        weekly_low = np.full(len(high_1d), np.nan)
        weekly_close = np.full(len(high_1d), np.nan)
        
        for i in range(4, len(high_1d)):
            weekly_high[i] = np.max(high_1d[i-4:i+1])
            weekly_low[i] = np.min(low_1d[i-4:i+1])
            weekly_close[i] = close_1d[i]  # Using current day's close as weekly close proxy
        
        # Weekly pivot point
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        weekly_r1 = 2 * weekly_pivot - weekly_low
        weekly_s1 = 2 * weekly_pivot - weekly_high
        weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
        weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
        
        # Align weekly levels to 6h timeframe
        weekly_pivot_6h = align_htf_to_ltf(prices, df_1d, weekly_pivot)
        weekly_r2_6h = align_htf_to_ltf(prices, df_1d, weekly_r2)
        weekly_s2_6h = align_htf_to_ltf(prices, df_1d, weekly_s2)
    else:
        return np.zeros(n)
    
    # Volume spike detection (20-period average on 6h)
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any critical data is NaN
        if (np.isnan(weekly_pivot_6h[i]) or 
            np.isnan(weekly_r2_6h[i]) or
            np.isnan(weekly_s2_6h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 6h volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.0
        
        if position == 0:
            # Long: Price breaks above weekly R2 with volume confirmation
            if (close[i] > weekly_r2_6h[i] and 
                volume_ratio > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below weekly S2 with volume confirmation
            elif (close[i] < weekly_s2_6h[i] and 
                  volume_ratio > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below weekly pivot
            if close[i] < weekly_pivot_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above weekly pivot
            if close[i] > weekly_pivot_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_WeeklyPivot_R2S2_Breakout"
timeframe = "6h"
leverage = 1.0