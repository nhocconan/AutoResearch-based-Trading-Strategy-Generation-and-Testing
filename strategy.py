#!/usr/bin/env python3
"""
12h_1W_WK1_WK2_WK3_WK4_Rotation
Hypothesis: Uses weekly pivot levels (WK1-WK4) on 1w timeframe. Trades breakouts of these levels
in the direction of the weekly trend (above/below weekly pivot) with volume confirmation.
Designed for both bull and bear markets by filtering with weekly trend. Target: 12-37 trades/year on 12h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for multi-timeframe analysis
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot and support/resistance levels
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    wk1 = pivot_1w * 2 - low_1w          # R1
    wk2 = pivot_1w + (high_1w - low_1w)  # R2
    wk3 = high_1w + 2 * (pivot_1w - low_1w)  # R3
    wk4 = 3 * pivot_1w - 2 * low_1w      # R4 (or S4 equivalent)
    sk1 = pivot_1w * 2 - high_1w         # S1
    sk2 = pivot_1w - (high_1w - low_1w)  # S2
    sk3 = low_1w - 2 * (high_1w - pivot_1w)  # S3
    sk4 = 3 * pivot_1w - 2 * high_1w     # S4
    
    # Align all levels to 12h timeframe (wait for bar close)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    wk1_aligned = align_htf_to_ltf(prices, df_1w, wk1)
    wk2_aligned = align_htf_to_ltf(prices, df_1w, wk2)
    wk3_aligned = align_htf_to_ltf(prices, df_1w, wk3)
    wk4_aligned = align_htf_to_ltf(prices, df_1w, wk4)
    sk1_aligned = align_htf_to_ltf(prices, df_1w, sk1)
    sk2_aligned = align_htf_to_ltf(prices, df_1w, sk2)
    sk3_aligned = align_htf_to_ltf(prices, df_1w, sk3)
    sk4_aligned = align_htf_to_ltf(prices, df_1w, sk4)
    
    # Volume confirmation: current volume > 2.0 x 24-period average (more selective)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(wk1_aligned[i]) or np.isnan(wk2_aligned[i]) or
            np.isnan(wk3_aligned[i]) or np.isnan(wk4_aligned[i]) or np.isnan(sk1_aligned[i]) or
            np.isnan(sk2_aligned[i]) or np.isnan(sk3_aligned[i]) or np.isnan(sk4_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above both WK2 and above weekly pivot, with volume
            if (close[i] > wk2_aligned[i] and 
                close[i] > pivot_1w_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below both SK2 and below weekly pivot, with volume
            elif (close[i] < sk2_aligned[i] and 
                  close[i] < pivot_1w_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns to weekly pivot or breaks below SK2
            if (not np.isnan(pivot_1w_aligned[i]) and close[i] < pivot_1w_aligned[i]) or \
               (not np.isnan(sk2_aligned[i]) and close[i] < sk2_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to weekly pivot or breaks above WK2
            if (not np.isnan(pivot_1w_aligned[i]) and close[i] > pivot_1w_aligned[i]) or \
               (not np.isnan(wk2_aligned[i]) and close[i] > wk2_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1W_WK1_WK2_WK3_WK4_Rotation"
timeframe = "12h"
leverage = 1.0