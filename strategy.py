#!/usr/bin/env python3
# 6h_Monthly_Pivot_Reversion_with_Volume
# Hypothesis: Combines monthly pivot levels with mean reversion at support/resistance and volume confirmation.
# Uses monthly pivot S2/R2 for mean reversion entries and monthly pivot S3/R3 for breakout continuation.
# Volume filter ensures participation. Designed to work in both trending and ranging markets by capturing
# reversals at key monthly levels and breakouts with confirmation. Target: 15-25 trades/year per symbol.

name = "6h_Monthly_Pivot_Reversion_with_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Get monthly data for pivot levels
    df_1m = get_htf_data(prices, '1M')
    if len(df_1m) < 2:
        return np.zeros(n)
    
    high_1m = df_1m['high'].values
    low_1m = df_1m['low'].values
    close_1m = df_1m['close'].values
    
    # Calculate monthly pivot points and support/resistance levels
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    pivot = (high_1m + low_1m + close_1m) / 3.0
    r1 = 2 * pivot - low_1m
    s1 = 2 * pivot - high_1m
    r2 = pivot + (high_1m - low_1m)
    s2 = pivot - (high_1m - low_1m)
    r3 = high_1m + 2 * (pivot - low_1m)
    s3 = low_1m - 2 * (high_1m - pivot)
    
    # Align monthly levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1m, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1m, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1m, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1m, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1m, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1m, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1m, s3)
    
    # Volume filter: 6h volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(s2_aligned[i]) or np.isnan(r2_aligned[i]) or \
           np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or \
           np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long at S2 with volume confirmation (mean reversion)
            if close[i] <= s2_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short at R2 with volume confirmation (mean reversion)
            elif close[i] >= r2_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
            # Enter long on break above R3 with volume (breakout continuation)
            elif close[i] > r3_aligned[i] and volume_ratio[i] > 2.0:
                signals[i] = 0.25
                position = 1
            # Enter short on break below S3 with volume (breakout continuation)
            elif close[i] < s3_aligned[i] and volume_ratio[i] > 2.0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reaches R1 (take profit) or breaks below S2 (stop)
            if close[i] >= r1_aligned[i] or close[i] < s2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches S1 (take profit) or breaks above R2 (stop)
            if close[i] <= s1_aligned[i] or close[i] > r2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals