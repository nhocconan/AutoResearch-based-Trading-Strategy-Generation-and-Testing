#!/usr/bin/env python3
# [24896] 12h_1d_camarilla_pivot_v1
# Hypothesis: 12-hour Camarilla pivot levels from 1-day data with volume confirmation and chop regime filter.
# Long when price touches S1 or S2 support with volume > 1.3x average and chop > 61.8 (range).
# Short when price touches R1 or R2 resistance with volume > 1.3x average and chop > 61.8 (range).
# Exit when price reaches opposite pivot level (S3/R3) or chop < 38.2 (trend).
# Uses tight entry conditions to limit trades (~20-30/year) and reduce fee drag.
# Designed to work in ranging markets (2022-2024) and avoid trending whipsaws.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_v1"
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
    
    # Get 1-day data for Camarilla pivots and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = np.full(len(df_1d), np.nan)
    camarilla_r2 = np.full(len(df_1d), np.nan)
    camarilla_r1 = np.full(len(df_1d), np.nan)
    camarilla_s1 = np.full(len(df_1d), np.nan)
    camarilla_s2 = np.full(len(df_1d), np.nan)
    camarilla_s3 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i < 1:  # Need previous day
            continue
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        rang = ph - pl
        camarilla_r3[i] = pc + rang * 1.1 / 2
        camarilla_r2[i] = pc + rang * 1.1 / 4
        camarilla_r1[i] = pc + rang * 1.1 / 6
        camarilla_s1[i] = pc - rang * 1.1 / 6
        camarilla_s2[i] = pc - rang * 1.1 / 4
        camarilla_s3[i] = pc - rang * 1.1 / 2
    
    # Calculate 1-day Choppiness Index (14-period)
    chop = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        tr = np.zeros(len(df_1d))
        atr = np.zeros(len(df_1d))
        for i in range(1, len(df_1d)):
            tr[i] = max(
                high_1d[i] - low_1d[i],
                abs(high_1d[i] - close_1d[i-1]),
                abs(low_1d[i] - close_1d[i-1])
            )
        for i in range(14, len(df_1d)):
            atr[i] = np.mean(tr[i-13:i+1])
            if atr[i] > 0:
                chop[i] = 100 * np.log10(sum(tr[i-13:i+1]) / (atr[i] * 14)) / np.log10(14)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1-day indicators to 12-hour timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        
        if position == 1:  # Long
            # Exit: price reaches S3 or chop < 38.2 (trend)
            if price <= camarilla_s3_aligned[i] or chop_aligned[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price reaches R3 or chop < 38.2 (trend)
            if price >= camarilla_r3_aligned[i] or chop_aligned[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price touches S1 or S2 with volume expansion and chop > 61.8 (range)
            if (abs(price - camarilla_s1_aligned[i]) < 0.001 * price or 
                abs(price - camarilla_s2_aligned[i]) < 0.001 * price) and \
               vol_ratio > 1.3 and chop_aligned[i] > 61.8:
                position = 1
                signals[i] = 0.25
            # Enter short: price touches R1 or R2 with volume expansion and chop > 61.8 (range)
            elif (abs(price - camarilla_r1_aligned[i]) < 0.001 * price or 
                  abs(price - camarilla_r2_aligned[i]) < 0.001 * price) and \
                 vol_ratio > 1.3 and chop_aligned[i] > 61.8:
                position = -1
                signals[i] = -0.25
    
    return signals