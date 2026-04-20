#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_Volume_Filter_v1
Concept: Use 1d Camarilla levels for 1h breakout entries with volume confirmation.
- Long: Price breaks above R1 with volume > 1.5x 20-period average
- Short: Price breaks below S1 with volume > 1.5x 20-period average
- Exit: Price returns to Pivot point
- Position sizing: 0.20
- Session filter: 08-20 UTC only
- Target: 15-30 trades/year (60-120 total over 4 years)
Works in bull/bear: Camarilla levels adapt to volatility, volume confirms breakout strength
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R1S1_Breakout_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Calculate 1d Camarilla levels ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: Range = (High - Low)
    # R1 = Close + (Range * 1.1/12)
    # S1 = Close - (Range * 1.1/12)
    # Pivot = (High + Low + Close) / 3
    rng = high_1d - low_1d
    r1 = close_1d + (rng * 1.1 / 12)
    s1 = close_1d - (rng * 1.1 / 12)
    pivot = (high_1d + low_1d + close_1d) / 3
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # === 1h: Volume confirmation ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === Session filter: 08-20 UTC only ===
    hours = prices.index.hour  # Already datetime64[ms], .hour works
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after volume MA warmup
        # Session filter
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any value is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        pivot_val = pivot_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation
            if price > r1_val and vol_ratio_val > 1.5:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below S1 with volume confirmation
            elif price < s1_val and vol_ratio_val > 1.5:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to Pivot point
            if price <= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Price returns to Pivot point
            if price >= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals