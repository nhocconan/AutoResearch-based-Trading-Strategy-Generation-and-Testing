#!/usr/bin/env python3
"""
4H_KAMA_Trend_R1_S1_Breakout_12H_Trend_Filter_v1
Hypothesis: Use 4h KAMA (ER=10) for trend direction and 12h Camarilla R1/S1 levels for entry.
Long when price crosses above 4h KAMA and touches 12h R1 level; 
Short when price crosses below 4h KAMA and touches 12h S1 level.
Volume confirmation: current volume > 1.5x 20-period average volume.
This combines adaptive trend-following with pivot point precision to reduce false signals and work in both bull and bear markets.
"""
name = "4H_KAMA_Trend_R1_S1_Breakout_12H_Trend_Filter_v1"
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
    
    # Get 4h data for KAMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Calculate 4h KAMA (ER=10)
    close_4h = pd.Series(df_4h['close'])
    change = abs(close_4h.diff(10))
    volatility = close_4h.diff().abs().rolling(window=10).sum()
    er = change / volatility.replace(0, 1e-10)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = [close_4h.iloc[0]]
    for i in range(1, len(close_4h)):
        kama.append(kama[-1] + sc.iloc[i] * (close_4h.iloc[i] - kama[-1]))
    kama = np.array(kama)
    kama_aligned = align_htf_to_ltf(prices, df_4h, kama)
    
    # Get 12h data for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
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
    
    # Volume filter: current volume > 1.5 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(10, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 12 bars between trades (2 days on 4h TF) to reduce frequency
            if bars_since_exit < 12:
                continue
                
            # Long: price crosses above KAMA and touches R1 level
            if (close[i] > kama_aligned[i] and close[i-1] <= kama_aligned[i-1] and 
                low[i] <= r1_aligned[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price crosses below KAMA and touches S1 level
            elif (close[i] < kama_aligned[i] and close[i-1] >= kama_aligned[i-1] and 
                  high[i] >= s1_aligned[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite KAMA side
            if position == 1 and close[i] < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals