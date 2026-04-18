#!/usr/bin/env python3
"""
6h_1d_Camarilla_Pivot_Reversal
Hypothesis: Use daily Camarilla pivot levels (R3/S3) for mean reversion in 6h timeframe. 
In ranging markets (common in 2025+), price tends to revert from extreme Camarilla levels (R3/S3). 
Add volume confirmation to avoid false signals. Works in both bull/bear markets as it fades extremes.
Target: 20-40 trades/year by requiring price at R3/S3 + volume > 1.5x average + reversal confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # S4 = close - 1.5 * (high - low)
    camarilla_r3 = np.full_like(close_1d, np.nan)
    camarilla_s3 = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        if i == 0:
            # For first day, use same day's high/low (no look-ahead)
            camarilla_r3[i] = close_1d[i] + 1.1 * (high_1d[i] - low_1d[i])
            camarilla_s3[i] = close_1d[i] - 1.1 * (high_1d[i] - low_1d[i])
        else:
            camarilla_r3[i] = close_1d[i-1] + 1.1 * (high_1d[i-1] - low_1d[i-1])
            camarilla_s3[i] = close_1d[i-1] - 1.1 * (high_1d[i-1] - low_1d[i-1])
    
    # Align Camarilla levels to 6h timeframe (wait for day close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price at or below S3 with volume and reversal confirmation
            if (close[i] <= s3_aligned[i] and vol_confirm[i] and 
                i > 0 and close[i] > close[i-1]):  # reversal confirmation
                signals[i] = 0.25
                position = 1
            # Short entry: price at or above R3 with volume and reversal confirmation
            elif (close[i] >= r3_aligned[i] and vol_confirm[i] and 
                  i > 0 and close[i] < close[i-1]):  # reversal confirmation
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price reaches midpoint (mean reversion target) or breaks below S3 (stop)
            mid_point = (r3_aligned[i] + s3_aligned[i]) / 2
            if (close[i] >= mid_point or 
                close[i] < s3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches midpoint or breaks above R3 (stop)
            mid_point = (r3_aligned[i] + s3_aligned[i]) / 2
            if (close[i] <= mid_point or 
                close[i] > r3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_Camarilla_Pivot_Reversal"
timeframe = "6h"
leverage = 1.0