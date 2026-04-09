#!/usr/bin/env python3
# 12h_camarilla_pivot_volume_v1
# Hypothesis: Uses Camarilla pivot levels on 1d timeframe with volume confirmation on 12h timeframe.
# Long when price touches S3 level with volume > 1.5x average; short when price touches R3 level with volume > 1.5x average.
# Designed to work in both bull and bear markets by capturing reversals at key pivot levels.
# Target: 15-25 trades/year (60-100 total over 4 years) with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    s3 = pivot - (range_1d * 1.1 / 2)
    s2 = pivot - (range_1d * 1.1 / 4)
    s1 = pivot - (range_1d * 1.1 / 6)
    r1 = pivot + (range_1d * 1.1 / 6)
    r2 = pivot + (range_1d * 1.1 / 4)
    r3 = pivot + (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    
    # Volume confirmation - 20 period average on 12h
    vol_ma_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 1:  # Long position
            # Exit: price moves above S1 level or volume confirmation fails
            if close[i] > s1_aligned[i] or not vol_ok:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves below R1 level or volume confirmation fails
            if close[i] < r1_aligned[i] or not vol_ok:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price touches or goes below S3 with volume confirmation
            if close[i] <= s3_aligned[i] and vol_ok:
                position = 1
                signals[i] = 0.25
            # Enter short: price touches or goes above R3 with volume confirmation
            elif close[i] >= r3_aligned[i] and vol_ok:
                position = -1
                signals[i] = -0.25
    
    return signals