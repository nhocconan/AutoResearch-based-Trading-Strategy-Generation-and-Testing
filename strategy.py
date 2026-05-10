#!/usr/bin/env python3
"""
12H_Volume_Regime_Filtered_Breakout
Hypothesis: Combines price breakouts above/below 20-period high/low on 12h chart
with volume confirmation (>1.5x 20-period average) and a chop regime filter
(CHOP > 61.8 = range, only trade breakouts when CHOP < 38.2 = trending).
Designed for low-frequency, high-conviction trades in both bull and bear markets.
Target: 15-30 trades/year, position size 0.25.
"""

name = "12H_Volume_Regime_Filtered_Breakout"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Choppiness Index (14-period)
    atr = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((atr * 14) / (highest_high - lowest_low)) / np.log10(14)
    
    # Volume filter: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(chop[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        is_trending = chop[i] < 38.2
        
        if position == 0:
            # Long entry: break above 20-period high with volume spike in trending market
            if (close[i] > high_max[i] and 
                volume[i] > vol_threshold[i] and 
                is_trending):
                signals[i] = 0.25
                position = 1
            # Short entry: break below 20-period low with volume spike in trending market
            elif (close[i] < low_min[i] and 
                  volume[i] > vol_threshold[i] and 
                  is_trending):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below 20-period low or trend ends
            if (close[i] < low_min[i] or 
                chop[i] >= 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above 20-period high or trend ends
            if (close[i] > high_max[i] or 
                chop[i] >= 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals