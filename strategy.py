#!/usr/bin/env python3
"""
6h Weekly Pivot Point Breakout with Volume Spike
Hypothesis: Weekly pivot points (R1/R2/S1/S2) act as significant support/resistance.
Breakouts above R1 or below S1 with volume confirmation capture momentum.
Weekly timeframe provides stable levels that work in both bull and bear markets.
Volume spike filter reduces false breakouts.
Target: 15-35 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (once before loop)
    df_w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points
    # P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    weekly_high = df_w['high'].values
    weekly_low = df_w['low'].values
    weekly_close = df_w['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_w, s2)
    
    # Volume spike: 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        pp = pivot_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        r2 = r2_aligned[i]
        s2 = s2_aligned[i]
        
        if position == 0:
            # Long: break above R1 with volume spike
            if price > r1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike
            elif price < s1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to pivot or breaks above R2 (take profit)
            if price <= pp or price >= r2:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to pivot or breaks below S2 (take profit)
            if price >= pp or price <= s2:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_R1S1_Breakout_Volume"
timeframe = "6h"
leverage = 1.0