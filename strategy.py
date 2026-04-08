#!/usr/bin/env python3
# 4h_1d_camarilla_volume_reversion_v1
# Hypothesis: 4-hour mean reversion at Camarilla pivot levels (S3/S4 for long, R3/R4 for short)
# with 1-day volume confirmation and chop regime filter. Works in both bull and bear markets
# by fading extremes in ranging conditions while avoiding strong trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_volume_reversion_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Camarilla pivot and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1-day volume moving average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    for i in range(20, len(vol_1d)):
        vol_ma_1d[i] = np.mean(vol_1d[i-20:i])
    
    # Calculate Camarilla pivot levels from previous day
    # Based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_s3 = np.full(len(close_1d), np.nan)
    camarilla_s4 = np.full(len(close_1d), np.nan)
    camarilla_r3 = np.full(len(close_1d), np.nan)
    camarilla_r4 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        # Use previous day's data
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        rang = ph - pl
        
        camarilla_s3[i] = pc - (1.1 * rang / 6)
        camarilla_s4[i] = pc - (1.1 * rang / 4)
        camarilla_r3[i] = pc + (1.1 * rang / 6)
        camarilla_r4[i] = pc + (1.1 * rang / 4)
    
    # Calculate 4-hour Choppiness Index (14-period) for regime filter
    def true_range(h, l, c_prev):
        return np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr_14 = np.full(n, np.nan)
    for i in range(14, n):
        atr_14[i] = np.mean(tr[i-14:i])
    
    chop = np.full(n, np.nan)
    for i in range(14, n):
        sum_tr = np.sum(tr[i-14:i])
        max_h = np.max(high[i-14:i])
        min_l = np.min(low[i-14:i])
        if max_h > min_l and sum_tr > 0:
            chop[i] = 100 * np.log10(sum_tr / (max_h - min_l)) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    # Align 1-day data to 4-hour timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(vol_ma_1d_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(chop[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma_1d_aligned[i] if vol_ma_1d_aligned[i] > 0 else 0
        price = close[i]
        
        # Chop regime filter: only trade when chop > 50 (ranging market)
        if chop[i] <= 50:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long
            # Exit: price reaches S3 or volume drops
            if price <= camarilla_s3_aligned[i] or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price reaches R3 or volume drops
            if price >= camarilla_r3_aligned[i] or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price touches/below S4 with volume confirmation
            if price <= camarilla_s4_aligned[i] and vol_ratio > 1.8:
                position = 1
                signals[i] = 0.25
            # Enter short: price touches/above R4 with volume confirmation
            elif price >= camarilla_r4_aligned[i] and vol_ratio > 1.8:
                position = -1
                signals[i] = -0.25
    
    return signals