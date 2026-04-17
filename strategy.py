#!/usr/bin/env python3
"""
6h Weekly Pivot Reversal with Volume Confirmation
Hypothesis: Price reverses at weekly pivot levels (R1/S1, R2/S2) on 6h timeframe with volume confirmation.
Works in bull/bear: Mean reversion at extreme weekly levels, with volume filter to avoid false signals.
Targets 15-30 trades/year per symbol.
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    weekly_high = df_1w['high'].shift(1)  # Prior week's high
    weekly_low = df_1w['low'].shift(1)    # Prior week's low
    weekly_close = df_1w['close'].shift(1) # Prior week's close
    
    pivot_point = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot_point - weekly_low
    s1 = 2 * pivot_point - weekly_high
    r2 = pivot_point + (weekly_high - weekly_low)
    s2 = pivot_point - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1.values)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2.values)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2.values)
    
    # Volume confirmation: 6h volume > 1.5x 24-period MA
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean()
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 24  # warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma[i]
        
        # Volume filter: require volume > 1.5x average
        volume_ok = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long reversal signals: price at or below S1/S2 with volume
            if price <= s1_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            elif price <= s2_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short reversal signals: price at or above R1/R2 with volume
            elif price >= r1_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
            elif price >= r2_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches pivot point or R1
            if price >= pivot_point[i] or price >= r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches pivot point or S1
            if price <= pivot_point[i] or price <= s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Reversal_Volume"
timeframe = "6h"
leverage = 1.0