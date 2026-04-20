#!/usr/bin/env python3
"""
6h_1d_Camarilla_Pivot_R1S1_Breakout_Volume_Conservative
Concept: 6h breakout above/below 1d Camarilla R1/S1 with volume confirmation.
- Long: Close > R1 AND volume > 1.5x 20-period volume average
- Short: Close < S1 AND volume > 1.5x 20-period volume average
- Exit: Close crosses back below R1 (long) or above S1 (short)
- Position sizing: 0.25
- Target: 15-30 trades/year (60-120 total over 4 years)
- Works in bull/bear: Camarilla levels adapt to volatility, volume confirms genuine breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Camarilla_Pivot_R1S1_Breakout_Volume_Conservative"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # === 6h: Volume average for confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Daily: Camarilla pivot levels ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    
    # Camarilla levels
    r1 = close_1d + (range_hl * 1.1 / 12)
    s1 = close_1d - (range_hl * 1.1 / 12)
    
    # Align to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Get values
        close_price = prices['close'].iloc[i]
        vol_ma_val = vol_ma[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ma_val) or np.isnan(r1_val) or np.isnan(s1_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close above R1 with volume confirmation
            if close_price > r1_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Close below S1 with volume confirmation
            elif close_price < s1_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close crosses below R1
            if close_price < r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close crosses above S1
            if close_price > s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals