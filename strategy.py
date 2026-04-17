#!/usr/bin/env python3
"""
6h Weekly Pivot + Price Channel Breakout
Long: Price breaks above weekly R2 with volume confirmation
Short: Price breaks below weekly S2 with volume confirmation
Exit: Price returns to weekly pivot or opposite breakout
Uses weekly pivot levels as structural support/resistance, effective in both trending and ranging markets.
Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def weekly_pivot(high, low, close):
    """Calculate weekly pivot points: P, R1, R2, S1, S2"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, r2, s1, s2

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 5:
        return np.zeros(n)
    
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly pivot levels
    pivot_w, r1_w, r2_w, s1_w, s2_w = weekly_pivot(high_w, low_w, close_w)
    
    # Align weekly levels to 6h timeframe (properly delayed for weekly bar close)
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_w, r2_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_w, s2_w)
    
    # Volume filter: 2x 6h volume SMA(20)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 20  # need volume SMA
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_w_aligned[i]) or np.isnan(r2_w_aligned[i]) or 
            np.isnan(s2_w_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_20[i]
        pivot = pivot_w_aligned[i]
        r2 = r2_w_aligned[i]
        s2 = s2_w_aligned[i]
        
        if position == 0:
            # Long: Price breaks above weekly R2 with volume confirmation
            if price > r2 and vol > 2.0 * vol_sma_val:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S2 with volume confirmation
            elif price < s2 and vol > 2.0 * vol_sma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to weekly pivot or breaks below S2
            if price < pivot or price < s2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to weekly pivot or breaks above R2
            if price > pivot or price > r2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R2S2_Breakout_Volume"
timeframe = "6h"
leverage = 1.0