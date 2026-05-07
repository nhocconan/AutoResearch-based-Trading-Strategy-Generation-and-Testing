#!/usr/bin/env python3
"""
12H_Camarilla_R1_S1_Breakout_1dTrend_Filter_v1
Hypothesis: Use 12h price action with 1d trend filter for higher timeframe stability.
Long when 12h price breaks above Camarilla R1 level with volume confirmation and 1d EMA50 uptrend.
Short when 12h price breaks below Camarilla S1 level with volume confirmation and 1d EMA50 downtrend.
This focuses on high-probability breakouts with trend alignment to reduce false signals and work in both bull and bear markets.
Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
"""
name = "12H_Camarilla_R1_S1_Breakout_1dTrend_Filter_v1"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Get 12h data for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (R1, S1)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    pivot = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    r1 = pivot + (range_12h * 1.1 / 12)
    s1 = pivot - (range_12h * 1.1 / 12)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    
    # Volume filter: current volume > 1.3 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(50, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 24 bars between trades (2 days on 12h TF) to reduce frequency
            if bars_since_exit < 24:
                continue
                
            # Long: price breaks above R1 with volume and 1d uptrend
            if (close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1] and 
                volume_filter[i] and close[i] > ema_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price breaks below S1 with volume and 1d downtrend
            elif (close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1] and 
                  volume_filter[i] and close[i] < ema_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite Camarilla level
            if position == 1 and close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals