#!/usr/bin/env python3
# 12h_daily_pivot_reversal_volume_v2
# Hypothesis: 12h Camarilla pivot reversals with 1d volume confirmation and 1w chop regime filter.
# Long: price touches S3 pivot level with volume > 1.5x average AND 1w chop > 61.8 (ranging)
# Short: price touches R3 pivot level with volume > 1.5x average AND 1w chop > 61.8 (ranging)
# Exit: price reaches opposite pivot level (S1/R1) or chop < 38.2 (trending)
# Uses 12h primary timeframe with 1d HTF for pivots/volume and 1w HTF for chop filter.
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_pivot_reversal_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels on 1d data
    camarilla_S1 = np.full(len(df_1d), np.nan)
    camarilla_S2 = np.full(len(df_1d), np.nan)
    camarilla_S3 = np.full(len(df_1d), np.nan)
    camarilla_R1 = np.full(len(df_1d), np.nan)
    camarilla_R2 = np.full(len(df_1d), np.nan)
    camarilla_R3 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        high_1d = df_1d['high'].iloc[i]
        low_1d = df_1d['low'].iloc[i]
        close_1d = df_1d['close'].iloc[i]
        if not (np.isnan(high_1d) or np.isnan(low_1d) or np.isnan(close_1d)):
            pivot = (high_1d + low_1d + close_1d) / 3.0
            range_1d = high_1d - low_1d
            camarilla_S1[i] = close_1d - (range_1d * 1.1 / 6)
            camarilla_S2[i] = close_1d - (range_1d * 1.1 / 4)
            camarilla_S3[i] = close_1d - (range_1d * 1.1 / 2)
            camarilla_R1[i] = close_1d + (range_1d * 1.1 / 6)
            camarilla_R2[i] = close_1d + (range_1d * 1.1 / 4)
            camarilla_R3[i] = close_1d + (range_1d * 1.1 / 2)
    
    # Calculate 1d volume ratio (current vs 20-period average)
    vol_sma_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_sma_1d[i] = np.mean(df_1d['volume'].iloc[i-20:i])
    vol_ratio_1d = np.where(vol_sma_1d > 0, df_1d['volume'].values / vol_sma_1d, 0)
    
    # Get 1w data for chop regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Chopiness Index on 1w data (14-period)
    chop_1w = np.full(len(df_1w), np.nan)
    for i in range(14, len(df_1w)):
        atr_sum = 0
        for j in range(i-13, i+1):
            tr = max(df_1w['high'].iloc[j] - df_1w['low'].iloc[j],
                     abs(df_1w['high'].iloc[j] - df_1w['close'].iloc[j-1]),
                     abs(df_1w['low'].iloc[j] - df_1w['close'].iloc[j-1]))
            atr_sum += tr
        atr = atr_sum / 14
        max_high = np.max(df_1w['high'].iloc[i-13:i+1].values)
        min_low = np.min(df_1w['low'].iloc[i-13:i+1].values)
        if max_high != min_low:
            chop_1w[i] = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
        else:
            chop_1w[i] = 50  # neutral when no range
    
    # Align 1d data to 12h timeframe
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    camarilla_S2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S2)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_R2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R2)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # Align 1w chop to 12h timeframe
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        vol_r = vol_ratio_1d_aligned[i]
        chop = chop_1w_aligned[i]
        price = close[i]
        
        if np.isnan(vol_r) or np.isnan(chop):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        s1 = camarilla_S1_aligned[i]
        s2 = camarilla_S2_aligned[i]
        s3 = camarilla_S3_aligned[i]
        r1 = camarilla_R1_aligned[i]
        r2 = camarilla_R2_aligned[i]
        r3 = camarilla_R3_aligned[i]
        
        if np.isnan(s1) or np.isnan(s2) or np.isnan(s3) or np.isnan(r1) or np.isnan(r2) or np.isnan(r3):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            if price >= r1 or chop < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if price <= s1 or chop < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long: price touches S3 with volume confirmation in ranging market
            if price <= s3 and vol_r > 1.5 and chop > 61.8:
                position = 1
                signals[i] = 0.25
            # Short: price touches R3 with volume confirmation in ranging market
            elif price >= r3 and vol_r > 1.5 and chop > 61.8:
                position = -1
                signals[i] = -0.25
    
    return signals