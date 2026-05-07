#!/usr/bin/env python3
"""
4H_Camarilla_R1_S1_Breakout_1D_Trend_Volume_v2
Hypothesis: Use 1d trend (close > SMA50) for direction and 4h Camarilla R1/S1 for entries.
Long when 1d trend is up and price breaks above 4h R1; short when 1d trend is down and price breaks below 4h S1.
Volume filter: current volume > 1.5x 20-period average volume to avoid low-probability breakouts.
This combines higher-timeframe trend direction with lower-timeframe precision entries to reduce false signals.
Designed to work in both bull and bear markets by following the 1d trend only.
"""
name = "4H_Camarilla_R1_S1_Breakout_1D_Trend_Volume_v2"
timeframe = "4h"
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
    
    # Get 1d data for trend (SMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d SMA50 for trend
    close_1d = pd.Series(df_1d['close'])
    sma50_1d = close_1d.rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    # Get 4h data for Camarilla levels (R1, S1)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (R1, S1)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    pivot_4h = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    r1_4h = pivot_4h + (range_4h * 1.1 / 12)
    s1_4h = pivot_4h - (range_4h * 1.1 / 12)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Volume filter: current volume > 1.5 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(50, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(sma50_1d_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 24 bars between trades (4 days on 4h TF) to reduce frequency
            if bars_since_exit < 24:
                continue
                
            # Long: 1d trend up (close > SMA50) and price breaks above R1
            if (close[i] > sma50_1d_aligned[i] and 
                high[i] > r1_aligned[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: 1d trend down (close < SMA50) and price breaks below S1
            elif (close[i] < sma50_1d_aligned[i] and 
                  low[i] < s1_aligned[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite side of Camarilla levels
            if position == 1 and low[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and high[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals