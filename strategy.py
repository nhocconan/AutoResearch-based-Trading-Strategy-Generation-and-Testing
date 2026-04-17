#!/usr/bin/env python3
"""
6h Weekly Pivot Breakout with Volume Confirmation
Hypothesis: Weekly pivot levels act as strong support/resistance zones. 
Breakout above weekly R1/R2 with volume confirmation indicates bullish momentum.
Breakdown below weekly S1/S2 with volume confirmation indicates bearish momentum.
Uses weekly pivots for higher timeframe structure (1w) and 6h for entry timing.
Works in both bull and bear markets by trading breakouts in direction of trend.
Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlc

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    # P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    # R4 = 3*P - 2*L, S4 = 3*P - 2*H
    
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Weekly pivot calculation
    pivot_w = (high_w + low_w + close_w) / 3.0
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    r3_w = high_w + 2 * (pivot_w - low_w)
    s3_w = low_w - 2 * (high_w - pivot_w)
    r4_w = 3 * pivot_w - 2 * low_w
    s4_w = 3 * pivot_w - 2 * high_w
    
    # Align weekly pivot levels to 6h timeframe
    pivot_w_aligned = align_ltf_to_hlc(prices, df_w, pivot_w)
    r1_w_aligned = align_ltf_to_hlc(prices, df_w, r1_w)
    s1_w_aligned = align_ltf_to_hlc(prices, df_w, s1_w)
    r2_w_aligned = align_ltf_to_hlc(prices, df_w, r2_w)
    s2_w_aligned = align_ltf_to_hlc(prices, df_w, s2_w)
    r3_w_aligned = align_ltf_to_hlc(prices, df_w, r3_w)
    s3_w_aligned = align_ltf_to_hlc(prices, df_w, s3_w)
    r4_w_aligned = align_ltf_to_hlc(prices, df_w, r4_w)
    s4_w_aligned = align_ltf_to_hlc(prices, df_w, s4_w)
    
    # Volume filter: 6h volume > 1.5x 24-period SMA
    vol_sma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 24  # need volume SMA
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_w_aligned[i]) or np.isnan(r1_w_aligned[i]) or 
            np.isnan(s1_w_aligned[i]) or np.isnan(r2_w_aligned[i]) or 
            np.isnan(s2_w_aligned[i]) or np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma[i]
        
        # Get current weekly levels
        r1 = r1_w_aligned[i]
        s1 = s1_w_aligned[i]
        r2 = r2_w_aligned[i]
        s2 = s2_w_aligned[i]
        r3 = r3_w_aligned[i]
        s3 = s3_w_aligned[i]
        r4 = r4_w_aligned[i]
        s4 = s4_w_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume confirmation
            if price > r1 and vol > 1.5 * vol_sma_val:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume confirmation
            elif price < s1 and vol > 1.5 * vol_sma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls back below pivot or reversal signal
            if price < pivot_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above pivot or reversal signal
            if price > pivot_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Breakout_Volume"
timeframe = "6h"
leverage = 1.0