#!/usr/bin/env python3
"""
4h_4H_12H_Swing_Signal_Combo
Hypothesis: Combine 4h swing high/low with 12h momentum. Enter long when price breaks above 4h swing high AND 12h price > SMA50; short when price breaks below 4h swing low AND 12h price < SMA50. Exit on opposite swing break or 12h momentum reversal. Uses swing points for structure and higher timeframe for trend filter. Target: 80-150 trades over 4 years (20-38/year) with position size 0.25. Works in bull/bear: 12h SMA50 filter avoids counter-trend trades, swing breaks capture momentum shifts.
"""

name = "4h_4H_12H_Swing_Signal_Combo"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    sma50_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 50:
        sma50_12h[49] = np.mean(close_12h[:50])
        for i in range(50, len(close_12h)):
            sma50_12h[i] = (sma50_12h[i-1] * 49 + close_12h[i]) / 50
    sma50_12h_aligned = align_htf_to_ltf(prices, df_12h, sma50_12h)
    
    # Calculate 4h swing points (fractals)
    # Swing high: high > previous 2 highs AND next 2 highs
    # Swing low: low < previous 2 lows AND next 2 lows
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    
    for i in range(2, n-2):
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            swing_high[i] = True
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            swing_low[i] = True
    
    # Get swing levels (last swing high/low)
    swing_high_level = np.full(n, np.nan)
    swing_low_level = np.full(n, np.nan)
    last_high = np.nan
    last_low = np.nan
    
    for i in range(n):
        if swing_high[i]:
            last_high = high[i]
        if swing_low[i]:
            last_low = low[i]
        swing_high_level[i] = last_high
        swing_low_level[i] = last_low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 10  # Ensure some lookback
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(swing_high_level[i]) or np.isnan(swing_low_level[i]) or 
            np.isnan(sma50_12h_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above swing high AND 12h price > SMA50
            if close[i] > swing_high_level[i] and close[i] > sma50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below swing low AND 12h price < SMA50
            elif close[i] < swing_low_level[i] and close[i] < sma50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below swing low OR 12h price < SMA50
            if close[i] < swing_low_level[i] or close[i] < sma50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above swing high OR 12h price > SMA50
            if close[i] > swing_high_level[i] or close[i] > sma50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals