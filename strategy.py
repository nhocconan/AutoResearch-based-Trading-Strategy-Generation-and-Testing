#!/usr/bin/env python3
"""
6h_WeeklyPivot_R3S3_Reversal_S4S4_Breakout
Hypothesis: Weekly pivot levels act as strong support/resistance. Price rejection at R3/S3 (reversal zones) or breakout beyond R4/S4 (breakout zones) with volume confirmation captures mean reversion and trend continuation. Works in both bull and bear markets by using weekly structure and volume filter to avoid false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points using standard formula
    # P = (H + L + C) / 3
    # R3 = H + 2*(P - L)
    # S3 = L - 2*(H - P)
    # R4 = H + 3*(P - L)
    # S4 = L - 3*(H - P)
    high_1w = df_1w['high']
    low_1w = df_1w['low']
    close_1w = df_1w['close']
    
    pivot = (high_1w + low_1w + close_1w) / 3
    r3 = high_1w + 2 * (pivot - low_1w)
    s3 = low_1w - 2 * (high_1w - pivot)
    r4 = high_1w + 3 * (pivot - low_1w)
    s4 = low_1w - 3 * (high_1w - pivot)
    
    # Shift by 1 to use previous weekly bar's levels only
    r3_prev = r3.shift(1).values
    s3_prev = s3.shift(1).values
    r4_prev = r4.shift(1).values
    s4_prev = s4.shift(1).values
    
    # Align to 6h timeframe
    r3_aligned = align_ltf_to_htf(prices, df_1w, r3_prev)
    s3_aligned = align_ltf_to_htf(prices, df_1w, s3_prev)
    r4_aligned = align_ltf_to_htf(prices, df_1w, r4_prev)
    s4_aligned = align_ltf_to_htf(prices, df_1w, s4_prev)
    
    # Volume spike: 2x 20-period average on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        
        if position == 0:
            # Long reversal: rejection at S3 with volume spike
            if low[i] <= s3_val and close[i] > s3_val and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short reversal: rejection at R3 with volume spike
            elif high[i] >= r3_val and close[i] < r3_val and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            # Long breakout: break above R4 with volume spike
            elif close[i] > r4_val and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: break below S4 with volume spike
            elif close[i] < s4_val and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to S3 or breaks below S4
            if close[i] < s3_val or close[i] < s4_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to R3 or breaks above R4
            if close[i] > r3_val or close[i] > r4_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_R3S3_Reversal_S4S4_Breakout"
timeframe = "6h"
leverage = 1.0