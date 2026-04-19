#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
# Donchian captures breakouts with clear entry/exit levels
# Weekly pivot provides higher timeframe bias to avoid counter-trend trades
# Volume confirmation filters weak breakouts and confirms institutional participation
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries
name = "6h_Donchian20_1wPivot_Direction_Volume"
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
    
    # 1d data for weekly pivot calculation (need daily data to aggregate to weekly)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points from daily data
    # Group daily data into weeks (starting from first available date)
    high_weekly = []
    low_weekly = []
    close_weekly = []
    
    # Simple approach: use rolling window of 5 days for weekly approximation
    # More accurate would require actual week grouping, but 5-day rolling approximates weekly
    high_5d = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().values
    low_5d = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().values
    close_5d = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().values
    
    # Calculate pivot points for each 5-day period
    pivot = (high_5d + low_5d + close_5d) / 3.0
    r1 = 2 * pivot - low_5d
    s1 = 2 * pivot - high_5d
    r2 = pivot + (high_5d - low_5d)
    s2 = pivot - (high_5d - low_5d)
    r3 = high_5d + 2 * (pivot - low_5d)
    s3 = low_5d - 2 * (high_5d - pivot)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Donchian channel (20-period) on 6h
    period = 20
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper + above weekly pivot R1 + volume confirmation
            if (close[i] > highest_high[i] and 
                close[i] > r1_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + below weekly pivot S1 + volume confirmation
            elif (close[i] < lowest_low[i] and 
                  close[i] < s1_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below Donchian lower or below weekly pivot S1
            if (close[i] < lowest_low[i]) or (close[i] < s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above Donchian upper or above weekly pivot R1
            if (close[i] > highest_high[i]) or (close[i] > r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals