#!/usr/bin/env python3
"""
1d_Weekly_Donchian_Breakout_Volume_Filter_v1
Weekly Donchian(20) breakout with volume confirmation on 1d timeframe.
Long when price breaks above weekly Donchian high with volume > 1.5x average.
Short when price breaks below weekly Donchian low with volume > 1.5x average.
Exit when price crosses the weekly Donchian midpoint.
Designed to capture strong trends with volume confirmation, avoiding false breakouts.
Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly Donchian channels ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full_like(close_1w, np.nan)
    donchian_low = np.full_like(close_1w, np.nan)
    donchian_mid = np.full_like(close_1w, np.nan)
    
    for i in range(20, len(close_1w)):
        donchian_high[i] = np.max(high_1w[i-20:i])
        donchian_low[i] = np.min(low_1w[i-20:i])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2
    
    # Align to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Volume average (20-period)
    vol_avg = np.full(n, np.nan)
    for i in range(20, n):
        vol_avg[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 20
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume condition: current volume > 1.5x average
        volume_condition = volume[i] > 1.5 * vol_avg[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above weekly Donchian high with volume confirmation
            if (close[i] > donchian_high_aligned[i] and 
                volume_condition):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below weekly Donchian low with volume confirmation
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_condition):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below weekly Donchian midpoint
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above weekly Donchian midpoint
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Donchian_Breakout_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0