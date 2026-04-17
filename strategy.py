#!/usr/bin/env python3
"""
12h_1d_Camarilla_R1_S1_Breakout_VolumeFilter
Strategy: Camarilla pivot breakout with volume confirmation on 12h timeframe.
- Uses daily Camarilla pivot levels (R1, S1) for entries
- Long when price breaks above R1 with volume > 2x 20-period average
- Short when price breaks below S1 with volume > 2x 20-period average
- Exit when price returns to daily pivot point
- Position size: 0.25 for long, -0.25 for short
- Works in both bull and bear markets by capturing mean reversion at extreme levels
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # R1 = C + (H - L) * 1.1/12
    # S1 = C - (H - L) * 1.1/12
    # Pivot = (H + L + C) / 3
    range_1d = high_1d - low_1d
    r1_1d = close_1d + range_1d * 1.1 / 12
    s1_1d = close_1d - range_1d * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    
    # Align to 12h timeframe (using previous day's levels)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Volume confirmation (20-period average)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period average
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        if position == 0:
            # Long: break above R1 with volume
            if close[i] > r1_1d_aligned[i-1] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume
            elif close[i] < s1_1d_aligned[i-1] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: return to pivot point
            if close[i] > pivot_1d_aligned[i-1]:
                signals[i] = 0.25
            else:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Exit short: return to pivot point
            if close[i] < pivot_1d_aligned[i-1]:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_1d_Camarilla_R1_S1_Breakout_VolumeFilter"
timeframe = "12h"
leverage = 1.0