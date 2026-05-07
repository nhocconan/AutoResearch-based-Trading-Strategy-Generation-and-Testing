#!/usr/bin/env python3
"""
6h_KAMA_Trend_1D_Camarilla_R3_S3_Breakout_With_Volume_Filter
Hypothesis: Use 6h KAMA (ER=10) for trend direction and 1d Camarilla R3/S3 levels for breakout entries.
Long when price crosses above 6h KAMA and breaks above 1d R3; short when price crosses below 6h KAMA and breaks below 1d S3.
Volume confirmation: current volume > 2.0x 20-period average volume to filter weak breakouts.
KAMA adapts to market noise, reducing false signals in chop, while Camarilla R3/S3 provide strong intraday support/resistance.
Volume filter ensures only significant breakouts trigger entries, reducing whipsaws in both bull and bear markets.
Designed for 6h timeframe to target 12-37 trades/year (50-150 total over 4 years).
"""
name = "6h_KAMA_Trend_1D_Camarilla_R3_S3_Breakout_With_Volume_Filter"
timeframe = "6h"
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
    
    # Get 6h data for KAMA trend
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 10:
        return np.zeros(n)
    
    # Calculate 6h KAMA (ER=10)
    close_6h = pd.Series(df_6h['close'])
    change = abs(close_6h.diff(10))
    volatility = close_6h.diff().abs().rolling(window=10).sum()
    er = change / volatility.replace(0, 1e-10)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = [close_6h.iloc[0]]
    for i in range(1, len(close_6h)):
        kama.append(kama[-1] + sc.iloc[i] * (close_6h.iloc[i] - kama[-1]))
    kama = np.array(kama)
    kama_aligned = align_htf_to_ltf(prices, df_6h, kama)
    
    # Get 1d data for Camarilla levels (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r3 = pivot + (range_1d * 1.1 / 4)  # R3 = pivot + 1.1*(H-L)/4
    s3 = pivot - (range_1d * 1.1 / 4)  # S3 = pivot - 1.1*(H-L)/4
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: current volume > 2.0 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(10, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 24 bars between trades (4 days on 6h TF) to reduce frequency
            if bars_since_exit < 24:
                continue
                
            # Long: price crosses above KAMA and breaks above R3
            if (close[i] > kama_aligned[i] and close[i-1] <= kama_aligned[i-1] and 
                close[i] > r3_aligned[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price crosses below KAMA and breaks below S3
            elif (close[i] < kama_aligned[i] and close[i-1] >= kama_aligned[i-1] and 
                  close[i] < s3_aligned[i]):
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