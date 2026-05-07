#!/usr/bin/env python3
"""
1D_KAMA_Trend_1W_Camarilla_R1_S1_Breakout_v2
Hypothesis: Use daily KAMA (ER=10) for trend direction and weekly Camarilla R1/S1 levels for entry.
Long when price crosses above daily KAMA and touches weekly R1 level;
Short when price crosses below daily KAMA and touches weekly S1 level.
Volume confirmation: current volume > 1.5x 20-period average volume.
This strategy targets 1d timeframe with 1w HTF to reduce trade frequency and improve robustness.
"""
name = "1D_KAMA_Trend_1W_Camarilla_R1_S1_Breakout_v2"
timeframe = "1d"
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
    
    # Get daily data for KAMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate daily KAMA (ER=10)
    close_1d = pd.Series(df_1d['close'])
    change = abs(close_1d.diff(10))
    volatility = close_1d.diff().abs().rolling(window=10).sum()
    er = change / volatility.replace(0, 1e-10)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = [close_1d.iloc[0]]
    for i in range(1, len(close_1d)):
        kama.append(kama[-1] + sc.iloc[i] * (close_1d.iloc[i] - kama[-1]))
    kama = np.array(kama)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Get weekly data for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (R1, S1)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r1 = pivot + (range_1w * 1.1 / 12)
    s1 = pivot - (range_1w * 1.1 / 12)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
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
            # Minimum 5 days between trades to reduce frequency
            if bars_since_exit < 5:
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