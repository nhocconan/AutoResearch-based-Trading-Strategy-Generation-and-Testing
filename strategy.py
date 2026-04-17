#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1_S1_Breakout_Volume_Filter
Hypothesis: On 4h, enter long when price breaks above Camarilla R1 level with volume confirmation, short when breaks below S1 level with volume. Uses 1d Camarilla levels for structure, volume filter for confirmation, and avoids false breakouts. Designed for 20-30 trades/year to minimize fee drag and work in both bull/bear regimes via breakout logic.
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
    
    # === 1d data for Camarilla pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align Camarilla levels to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume filter: current volume > 1.5x 20-period average on 1d
    vol_avg20_1d = pd.Series(volume_1d := df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_avg20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg20_1d)
    
    signals = np.zeros(n)
    
    # Warmup: covers Camarilla calculation and volume average
    warmup = 30
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(vol_avg20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 1d volume
        vol_1d_current = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_filter = vol_1d_current > 1.5 * vol_avg20_1d_aligned[i]
        
        # Entry conditions
        if position == 0:
            # Long: price breaks above R1 with volume
            if close[i] > r1_1d_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1 with volume
            elif close[i] < s1_1d_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: reverse when price returns to pivot level
        elif position == 1:
            if close[i] < pivot_1d[i]:  # exit long when price returns below pivot
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if close[i] > pivot_1d[i]:  # exit short when price returns above pivot
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_R1_S1_Breakout_Volume_Filter"
timeframe = "4h"
leverage = 1.0