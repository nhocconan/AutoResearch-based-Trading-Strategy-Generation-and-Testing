#!/usr/bin/env python3
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
    
    # Get daily data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate daily pivot points (standard floor trader pivots)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    # R2 = P + (H - L), S2 = P - (H - L)
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    r3 = high_1d + 2 * (pivot - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot)
    
    # Align pivots to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike detection (volume > 1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_sum = np.nansum(volume[:20])
        vol_ma[19] = vol_sum / 20
        for i in range(20, n):
            vol_ma[i] = vol_ma[i-1] + (volume[i] - volume[i-20]) / 20
    volume_ratio = np.divide(volume, vol_ma, out=np.full_like(volume, np.nan), where=vol_ma!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need daily data and volume MA
    start_idx = max(20, 0)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(r3_6h[i]) or
            np.isnan(s3_6h[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume_ratio[i]
        
        if position == 0:
            # Long: Price breaks above R3 with volume spike
            if price > r3_6h[i] and vol_ratio > 1.5:
                signals[i] = size
                position = 1
            # Short: Price breaks below S3 with volume spike
            elif price < s3_6h[i] and vol_ratio > 1.5:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price falls back below R2 or volume dries up
            if price < r2_6h[i] or vol_ratio < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Price rises back above S2 or volume dries up
            if price > s2_6h[i] or vol_ratio < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6H_Pivot_R3S3_Breakout_Volume"
timeframe = "6h"
leverage = 1.0