#!/usr/bin/env python3
"""
6h_LongOnly_WeeklyTrend_Following_v1
Hypothesis: Long only strategy using weekly trend filter (price above weekly SMA50) and 6h Donchian(20) breakout with volume confirmation.
Weekly SMA50 provides a robust trend filter that works in both bull and bear markets by only allowing longs in uptrends and staying flat in downtrends/ranges.
6h Donchian breakout captures momentum bursts, and volume confirmation reduces false signals.
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
"""
name = "6h_LongOnly_WeeklyTrend_Following_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (SMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly SMA50 for trend filter
    sma_50_1w = pd.Series(df_1w['close']).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Get 6h data for Donchian channels
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate 6h Donchian(20) - upper and lower bands
    donch_high = pd.Series(df_6h['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_6h['low']).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_6h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_6h, donch_low)
    
    # Volume filter: current volume > 1.5 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(50, 20, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(sma_50_1w_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 24 bars between trades (4 days on 6h TF) to reduce frequency
            if bars_since_exit < 24:
                continue
                
            # Long: price above weekly SMA50 (uptrend) AND price breaks above 6h Donchian upper + volume filter
            if (close[i] > sma_50_1w_aligned[i] and 
                close[i] > donch_high_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
        elif position == 1:
            # Exit: price breaks below 6h Donchian lower OR price falls below weekly SMA50 (trend change)
            if (close[i] < donch_low_aligned[i] or close[i] < sma_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25
    
    return signals