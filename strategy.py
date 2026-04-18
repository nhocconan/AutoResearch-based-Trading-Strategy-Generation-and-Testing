#!/usr/bin/env python3
"""
1d_Weekly_Pivot_R1S1_Breakout_Volume_Filter
Strategy: Weekly Camarilla pivot levels (R1/S1) on 1d timeframe with volume confirmation.
Long: Price breaks above weekly R1 with volume > 1.5x average volume.
Short: Price breaks below weekly S1 with volume > 1.5x average volume.
Exit: Price returns to weekly pivot point (PP) or opposite signal.
Designed for 1d timeframe: ~10-20 trades/year per symbol (40-80 total over 4 years).
Works in bull/bear via pivot levels acting as dynamic support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 10:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels
    # Pivot Point (PP) = (H + L + C) / 3
    pp = (high_1w + low_1w + close_1w) / 3
    # R1 = C + (H - L) * 1.1 / 12
    r1 = close_1w + (high_1w - low_1w) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    s1 = close_1w - (high_1w - low_1w) * 1.1 / 12
    
    # Align weekly levels to daily timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume confirmation: 20-day average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume
            if close[i] > r1_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume
            elif close[i] < s1_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to PP or breaks below S1
            if close[i] <= pp_aligned[i] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to PP or breaks above R1
            if close[i] >= pp_aligned[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Pivot_R1S1_Breakout_Volume_Filter"
timeframe = "1d"
leverage = 1.0