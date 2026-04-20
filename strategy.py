#!/usr/bin/env python3
"""
12h_1d_Pivot_R1S1_Breakout_Volume_v1
Concept: 12h pivot breakout with volume confirmation using daily pivots.
- Long: Close > R1 AND volume > 1.8x 24-period average
- Short: Close < S1 AND volume > 1.8x 24-period average
- Exit: Price returns to pivot point (PP)
- Uses daily pivot levels from 1d timeframe
- Position sizing: 0.25
- Target: 50-150 total trades over 4 years (12-37/year)
- Works in bull/bear: Pivots adapt to market structure, volume filter reduces false signals
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Pivot_R1S1_Breakout_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # === 1d: Calculate Pivot Points (Standard) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot Point (PP) = (H + L + C) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    # R1 = 2*PP - L
    r1_1d = 2 * pp_1d - low_1d
    # S1 = 2*PP - H
    s1_1d = 2 * pp_1d - high_1d
    
    # Align pivot levels to 12h
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 12h: Indicators ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume: 24-period average (2 days worth of 12h data)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Get values
        pp_val = pp_1d_aligned[i]
        r1_val = r1_1d_aligned[i]
        s1_val = s1_1d_aligned[i]
        current_vol_ma = vol_ma[i]
        current_volume = volume[i]
        current_close = close[i]
        
        # Skip if any value is NaN
        if (np.isnan(pp_val) or np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(current_vol_ma)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.8x 24-period average
        vol_condition = current_volume > 1.8 * current_vol_ma
        
        if position == 0:
            # Long: close above R1 with volume confirmation
            if current_close > r1_val and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: close below S1 with volume confirmation
            elif current_close < s1_val and vol_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: return to pivot point
            if current_close < pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: return to pivot point
            if current_close > pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals