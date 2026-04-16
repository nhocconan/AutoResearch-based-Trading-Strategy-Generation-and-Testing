#!/usr/bin/env python3
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
    
    # === 6h data (primary) ===
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # === 1d data (HTF for pivot levels) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Classic Pivot Points (based on prior day) ===
    # P = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    # R1 = 2*P - L
    r1_1d = 2 * pivot_1d - low_1d
    # S1 = 2*P - H
    s1_1d = 2 * pivot_1d - high_1d
    # R2 = P + (H - L)
    r2_1d = pivot_1d + (high_1d - low_1d)
    # S2 = P - (H - L)
    s2_1d = pivot_1d - (high_1d - low_1d)
    # R3 = H + 2*(P - L)
    r3_1d = high_1d + 2 * (pivot_1d - low_1d)
    # S3 = L - 2*(H - P)
    s3_1d = low_1d - 2 * (high_1d - pivot_1d)
    
    # Align pivot levels (no extra delay needed as they're based on prior day)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # === 6h volume ratio for confirmation ===
    vol_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_6h = volume_6h / vol_ma_20_6h
    vol_ratio_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ratio_6h)
    
    signals = np.zeros(n)
    
    # Warmup: enough for pivot calculation and volume MA
    warmup = 30
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(r2_1d_aligned[i]) or 
            np.isnan(s2_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(vol_ratio_6h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        pivot = pivot_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        r2 = r2_1d_aligned[i]
        s2 = s2_1d_aligned[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        vol_ratio = vol_ratio_6h_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price reaches R2 or volume drops
            if price >= r2 or vol_ratio < 0.8:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price reaches S2 or volume drops
            if price <= s2 or vol_ratio < 0.8:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long setup: price breaks above R1 with volume, target R2
            if price > r1 and vol_ratio > 1.5:
                signals[i] = 0.25
                position = 1
                continue
            # Short setup: price breaks below S1 with volume, target S2
            elif price < s1 and vol_ratio > 1.5:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Pivot_R1_S1_Breakout_Volume"
timeframe = "6h"
leverage = 1.0