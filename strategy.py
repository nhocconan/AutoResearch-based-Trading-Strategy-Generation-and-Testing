#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_Pullback
4h strategy using daily Camarilla pivot levels (R1/S1) with volume confirmation and pullback entry.
- Long: Pullback to S1 after breakout above R1, volume > 1.5x 20-period average
- Short: Pullback to R1 after breakdown below S1, volume > 1.5x 20-period average
- Exit: Opposite pullback level or intraday reversal
Designed for ~20-30 trades/year per symbol (80-120 total over 4 years)
Works in bull markets (breakout continuation) and bear markets (breakdown continuation)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for each day
    # Based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla R1 and S1 levels
    r1 = close_1d + (range_1d * 1.1 / 12)
    s1 = close_1d - (range_1d * 1.1 / 12)
    
    # Align daily levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need enough for volume average
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Pullback conditions
        pullback_to_s1 = low[i] <= s1_aligned[i] and close[i] > s1_aligned[i]
        pullback_to_r1 = high[i] >= r1_aligned[i] and close[i] < r1_aligned[i]
        
        if position == 0:
            # Long: pullback to S1 with volume confirmation
            if pullback_to_s1 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: pullback to R1 with volume confirmation
            elif pullback_to_r1 and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: pullback to R1 or intraday reversal below S1
            if pullback_to_r1 or close[i] < s1_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: pullback to S1 or intraday reversal above R1
            if pullback_to_s1 or close[i] > r1_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_Pullback"
timeframe = "4h"
leverage = 1.0