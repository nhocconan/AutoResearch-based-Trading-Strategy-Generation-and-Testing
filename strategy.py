#!/usr/bin/env python3
"""
4H_Camarilla_R1_S1_Breakout_12H_Trend_Filter_v2
Hypothesis: Use 12h EMA50 trend filter combined with 4h Camarilla R1/S1 breakouts.
Long when price breaks above R1 with 12h EMA50 uptrend; Short when price breaks below S1 with 12h EMA50 downtrend.
Volume confirmation: current volume > 1.5x 20-period average volume.
Minimum 12 bars between trades to reduce frequency and avoid overtrading.
Designed to work in both bull and bear markets by following higher timeframe trend.
"""
name = "4H_Camarilla_R1_S1_Breakout_12H_Trend_Filter_v2"
timeframe = "4h"
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
    
    # Get 4h data for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (R1, S1)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    pivot = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    r1 = pivot + (range_4h * 1.1 / 12)
    s1 = pivot - (range_4h * 1.1 / 12)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = pd.Series(df_12h['close'])
    ema_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
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
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 12 bars between trades (2 days on 4h TF) to reduce frequency
            if bars_since_exit < 12:
                continue
                
            # Long: price breaks above R1 with 12h EMA50 uptrend
            if (close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1] and 
                ema_12h_aligned[i] > ema_12h_aligned[i-1] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price breaks below S1 with 12h EMA50 downtrend
            elif (close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1] and 
                  ema_12h_aligned[i] < ema_12h_aligned[i-1] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite Camarilla level (R1 for long, S1 for short)
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