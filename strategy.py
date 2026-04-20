#!/usr/bin/env python3
"""
12h_1d_Camarilla_R1S1_Breakout_Volume_Filter_v1
Concept: 12h Camarilla pivot levels (R1/S1) from prior 1d + volume spike + close confirmation.
- Long: Close > R1 and volume > 1.5x avg volume
- Short: Close < S1 and volume > 1.5x avg volume
- Exit: Close crosses back below R1 (long) or above S1 (short)
- Position sizing: 0.25
- Target: 15-25 trades/year (60-100 total over 4 years)
- Works in bull/bear: Camarilla levels adapt to volatility, volume confirms breakout strength
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_R1S1_Breakout_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Calculate Camarilla pivot levels from prior 1d OHLC ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    
    # Camarilla levels:
    # R1 = C + (H - L) * 1.0833
    # S1 = C - (H - L) * 1.0833
    r1 = close_1d + range_1d * 1.0833
    s1 = close_1d - range_1d * 1.0833
    
    # Align to 12h timeframe (use prior day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Volume spike filter (12h) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Get values
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ma_val = vol_ma[i]
        close_val = prices['close'].iloc[i]
        volume_val = volume[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(vol_ma_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition
        vol_spike = volume_val > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: Close above R1 and volume spike
            if close_val > r1_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Close below S1 and volume spike
            elif close_val < s1_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close crosses back below R1
            if close_val < r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close crosses back above S1
            if close_val > s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals