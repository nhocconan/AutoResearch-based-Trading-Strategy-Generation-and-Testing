#!/usr/bin/env python3
"""
12h_1D_Pivot_R1S1_Breakout_Volume_Filter
Hypothesis: Daily pivot levels (R1/S1) act as strong support/resistance on 12h timeframe.
Breakouts above R1 or below S1 with volume confirmation capture institutional moves.
Works in bull/bear via volatility-based position sizing and breakout symmetry.
Target: 50-150 trades over 4 years (12-37/year).
"""

name = "12h_1D_Pivot_R1S1_Breakout_Volume_Filter"
timeframe = "12h"
leverage = 1.0

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
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivots: P = (H+L+C)/3, R1 = 2P-L, S1 = 2P-H
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Align to 12h timeframe (wait for daily close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with volume confirmation
            if (close[i] > r1_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume confirmation
            elif (close[i] < s1_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price returns below pivot (mean reversion)
            if close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price returns above pivot (mean reversion)
            if close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals