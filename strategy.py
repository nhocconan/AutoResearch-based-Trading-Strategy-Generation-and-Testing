#!/usr/bin/env python3
"""
12h_1d_Pivot_R1_S1_Breakout_Volume_Filter_v1
Strategy: Short-only mean reversion at Camarilla S1 with volume spike and range filter.
- Short when price breaks below Camarilla S1 (1-day) with volume > 2x 20-period average
- Exit when price returns to Camarilla C (midpoint) or breaks above R1
- Designed to work in both bull and bear markets by capturing overextended moves
- Position size: -0.25
- Uses 12h timeframe as primary, 1d for Camarilla levels
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
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    # R1 = close + (high - low) * 1.1 / 12
    # S1 = close - (high - low) * 1.1 / 12
    # C = (high + low + close) / 3
    diff_1d = high_1d - low_1d
    R1_1d = close_1d + diff_1d * 1.1 / 12
    S1_1d = close_1d - diff_1d * 1.1 / 12
    C_1d = (high_1d + low_1d + close_1d) / 3
    
    # Align to 12h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    C_1d_aligned = align_htf_to_ltf(prices, df_1d, C_1d)
    
    # Volume confirmation (20-period MA on 12h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, -1: short
    
    start_idx = max(20, 20)  # Need 20 bars for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_1d_aligned[i]) or 
            np.isnan(S1_1d_aligned[i]) or 
            np.isnan(C_1d_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period average
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Breakdown below S1
        breakdown = close[i] < S1_1d_aligned[i-1]
        
        # Return to midpoint C
        return_to_midpoint = abs(close[i] - C_1d_aligned[i]) < 0.003 * close[i]  # within 0.3% of C
        
        # Break above R1 (stop loss for short)
        break_above_R1 = close[i] > R1_1d_aligned[i-1]
        
        if position == 0:
            # Short: breakdown below S1 + volume filter
            if breakdown and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == -1:
            # Exit short: return to midpoint or break above R1 (stop loss)
            if return_to_midpoint or break_above_R1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Pivot_R1_S1_Breakout_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0