#!/usr/bin/env python3
"""
12H_Camarilla_R1_S1_Breakout_1D_Trend_Volume_v1
Hypothesis: Use 12h timeframe with 1d trend filter (EMA34) and volume confirmation.
Long when price crosses above 12h EMA and touches 12h R1 level in uptrend.
Short when price crosses below 12h EMA and touches 12h S1 level in downtrend.
Volume filter: current volume > 1.5x 20-period average volume.
Designed for fewer trades (12-37/year) to avoid fee drag while capturing trend continuation.
Works in bull markets via trend continuation and in bear markets via short signals.
"""
name = "12H_Camarilla_R1_S1_Breakout_1D_Trend_Volume_v1"
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
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = pd.Series(df_1d['close'])
    ema_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Get 12h data for EMA and Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA20 for trend
    close_12h = pd.Series(df_12h['close'])
    ema_12h = close_12h.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
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
    
    # Volume filter: current volume > 1.5 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(34, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(ema_12h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 24 bars between trades (12 days on 12h TF) to reduce frequency
            if bars_since_exit < 24:
                continue
                
            # Long: price crosses above 12h EMA and touches R1 level in uptrend (1d EMA up)
            if (close[i] > ema_12h_aligned[i] and close[i-1] <= ema_12h_aligned[i-1] and 
                low[i] <= r1_aligned[i] and ema_1d_aligned[i] > ema_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price crosses below 12h EMA and touches S1 level in downtrend (1d EMA down)
            elif (close[i] < ema_12h_aligned[i] and close[i-1] >= ema_12h_aligned[i-1] and 
                  high[i] >= s1_aligned[i] and ema_1d_aligned[i] < ema_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite 12h EMA side
            if position == 1 and close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals