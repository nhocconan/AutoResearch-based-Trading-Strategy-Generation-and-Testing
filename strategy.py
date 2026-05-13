#!/usr/bin/env python3
name = "6h_Pivot_Reversal_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Pivot points from previous day
    # P = (H + L + C) / 3
    # S1 = 2*P - H
    # R1 = 2*P - L
    # S2 = P - (H - L)
    # R2 = P + (H - L)
    # S3 = H - 2*(H - P)
    # R3 = L + 2*(P - L)
    # We'll use previous day's high, low, close
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate daily pivot points
    prev_high = df_1d['high'].shift(1).values  # previous day's high
    prev_low = df_1d['low'].shift(1).values    # previous day's low
    prev_close = df_1d['close'].shift(1).values # previous day's close
    
    # Pivot point
    pp = (prev_high + prev_low + prev_close) / 3.0
    # Support and resistance levels
    s1 = 2 * pp - prev_high
    r1 = 2 * pp - prev_low
    s2 = pp - (prev_high - prev_low)
    r2 = pp + (prev_high - prev_low)
    s3 = prev_high - 2 * (prev_high - pp)
    r3 = prev_low + 2 * (pp - prev_low)
    
    # Align pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    
    # Volume filter: current volume > 1.3 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        if (np.isnan(pp_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_filter = volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 0:
            # LONG: price touches S1 or S2 with volume, expecting bounce
            if vol_filter and (close[i] <= s1_aligned[i] * 1.005 or close[i] <= s2_aligned[i] * 1.005):
                signals[i] = 0.25
                position = 1
            # SHORT: price touches R1 or R2 with volume, expecting rejection
            elif vol_filter and (close[i] >= r1_aligned[i] * 0.995 or close[i] >= r2_aligned[i] * 0.995):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price reaches R1 or shows weakness
            if close[i] >= r1_aligned[i] * 0.995 or close[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price reaches S1 or shows strength
            if close[i] <= s1_aligned[i] * 1.005 or close[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals