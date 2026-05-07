#!/usr/bin/env python3
"""
4H_Camarilla_R3S3_Breakout_1dEMA34_Volume_Spike_v1
Hypothesis: Use 1d EMA34 for trend direction and 4h Camarilla R3/S3 levels for entry.
Long when price crosses above 4h EMA and touches R3 level; 
Short when price crosses below 4h EMA and touches S3 level.
Volume confirmation: current volume > 2.0x 20-period average volume.
This combines trend-following with precision pivot entries to reduce false signals and work in both bull and bear markets.
"""
name = "4H_Camarilla_R3S3_Breakout_1dEMA34_Volume_Spike_v1"
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
    
    # Get 4h data for EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4h EMA(34)
    close_4h = pd.Series(df_4h['close'])
    ema_4h = close_4h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 4h data for Camarilla levels (R3, S3)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_vals = df_4h['close'].values
    pivot = (high_4h + low_4h + close_4h_vals) / 3
    range_4h = high_4h - low_4h
    r3 = pivot + (range_4h * 1.1 / 4)  # R3 level
    s3 = pivot - (range_4h * 1.1 / 4)  # S3 level
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    
    # Volume filter: current volume > 2.0 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(34, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 8 bars between trades (1.33 days on 4h TF) to reduce frequency
            if bars_since_exit < 8:
                continue
                
            # Long: price crosses above EMA and touches R3 level
            if (close[i] > ema_4h_aligned[i] and close[i-1] <= ema_4h_aligned[i-1] and 
                low[i] <= r3_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price crosses below EMA and touches S3 level
            elif (close[i] < ema_4h_aligned[i] and close[i-1] >= ema_4h_aligned[i-1] and 
                  high[i] >= s3_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite EMA side
            if position == 1 and close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals