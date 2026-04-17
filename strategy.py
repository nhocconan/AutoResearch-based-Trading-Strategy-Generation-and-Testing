#!/usr/bin/env python3
"""
6h Weekly Pivot + Volume Spike Strategy
Long: Price breaks above weekly R1 with volume > 1.5x volume SMA(20)
Short: Price breaks below weekly S1 with volume > 1.5x volume SMA(20)
Exit: Price crosses back below weekly pivot (long) or above weekly pivot (short)
Uses weekly pivot points for structure, volume for confirmation, 6h timeframe for lower frequency
Target: 12-30 trades/year per symbol (48-120 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points: (H+L+C)/3
    # Then R1 = 2*P - L, S1 = 2*P - H
    typical_price = (df_w['high'] + df_w['low'] + df_w['close']) / 3
    pivot = typical_price.values
    r1 = (2 * pivot) - df_w['low'].values
    s1 = (2 * pivot) - df_w['high'].values
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_w, s1)
    
    # Calculate 6h volume SMA(20) for volume filter
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 20  # Volume SMA needs 20 bars
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        
        if position == 0:
            # Long: Price breaks above weekly R1 with volume > 1.5x volume SMA
            if price > r1_val and vol > 1.5 * vol_sma_val:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S1 with volume > 1.5x volume SMA
            elif price < s1_val and vol > 1.5 * vol_sma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses back below weekly pivot
            if price < pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses back above weekly pivot
            if price > pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_VolumeBreakout"
timeframe = "6h"
leverage = 1.0