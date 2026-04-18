#!/usr/bin/env python3
"""
4h_MultiTF_Structure_Breakout_V1
Hypothesis: Combine daily price structure (higher highs/lows) with 4h breakouts and volume confirmation. 
Works in bull markets via upward structure breaks and in bear markets via breakdowns of downward structure.
Uses daily swing points to filter 4h breakouts, reducing false signals and trade frequency.
"""

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
    
    # Daily swing high/low (pivot points) for structure
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily swing points: swing high = higher high, swing low = lower low
    swing_high = np.full(len(high_1d), np.nan)
    swing_low = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d)):
        # Swing high: higher than previous and next bar's high
        if high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i+1] if i+1 < len(high_1d) else high_1d[i] > high_1d[i-1]:
            swing_high[i] = high_1d[i]
        # Swing low: lower than previous and next bar's low
        if low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i+1] if i+1 < len(low_1d) else low_1d[i] < low_1d[i-1]:
            swing_low[i] = low_1d[i]
    
    # Forward fill swing points to get structure levels
    swing_high_ff = pd.Series(swing_high).ffill().values
    swing_low_ff = pd.Series(swing_low).ffill().values
    
    # Align daily structure to 4h
    swing_high_aligned = align_htf_to_ltf(prices, df_1d, swing_high_ff)
    swing_low_aligned = align_htf_to_ltf(prices, df_1d, swing_low_ff)
    
    # 4h 20-period Donchian channel for breakouts
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 2)  # Ensure indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(swing_high_aligned[i]) or np.isnan(swing_low_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 4h breaks above Donchian high with volume and above daily swing low (bullish structure)
            if (high[i] > donchian_high[i] and vol_confirm[i] and 
                close[i] > swing_low_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: 4h breaks below Donchian low with volume and below daily swing high (bearish structure)
            elif (low[i] < donchian_low[i] and vol_confirm[i] and 
                  close[i] < swing_high_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below Donchian low or structure breaks down
            if (low[i] < donchian_low[i] or close[i] < swing_low_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above Donchian high or structure breaks up
            if (high[i] > donchian_high[i] or close[i] > swing_high_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_MultiTF_Structure_Breakout_V1"
timeframe = "4h"
leverage = 1.0