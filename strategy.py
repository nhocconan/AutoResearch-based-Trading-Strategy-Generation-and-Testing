#!/usr/bin/env python3
# 4h_daily_camarilla_pivot_volume_regime_v2
# Hypothesis: 4h Camarilla pivot levels from 1d + volume spike + 1d chop regime filter.
# Long: price touches S3 level with volume > 2.0x average AND 1d chop < 61.8 (trending)
# Short: price touches R3 level with volume > 2.0x average AND 1d chop < 61.8 (trending)
# Exit: price moves to opposite H4/L4 level or chop > 61.8 (ranging)
# Uses 4h primary timeframe with 1d HTF for pivot and regime filter.
# Target: 75-200 trades over 4 years (19-50/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_daily_camarilla_pivot_volume_regime_v2"
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
    
    # Get 1d data for Camarilla pivots and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = np.full(len(df_1d), np.nan)
    camarilla_s3 = np.full(len(df_1d), np.nan)
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    camarilla_h6 = np.full(len(df_1d), np.nan)
    camarilla_l6 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i == 0:
            camarilla_r3[i] = np.nan
            camarilla_s3[i] = np.nan
            camarilla_h4[i] = np.nan
            camarilla_l4[i] = np.nan
            camarilla_h6[i] = np.nan
            camarilla_l6[i] = np.nan
            continue
            
        # Use previous day's data for today's levels
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        
        pivot = (prev_high + prev_low + prev_close) / 3.0
        range_val = prev_high - prev_low
        
        camarilla_h4[i] = pivot + range_val * 1.1 / 2.0
        camarilla_l4[i] = pivot - range_val * 1.1 / 2.0
        camarilla_h6[i] = pivot + range_val * 1.1 / 4.0
        camarilla_l6[i] = pivot - range_val * 1.1 / 4.0
        camarilla_r3[i] = pivot + range_val * 1.1
        camarilla_s3[i] = pivot - range_val * 1.1
    
    # Calculate Chopiness Index on 1d data (14-period)
    chop_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_sum = 0
        for j in range(i-13, i+1):
            tr = max(high_1d[j] - low_1d[j],
                     abs(high_1d[j] - close_1d[j-1]),
                     abs(low_1d[j] - close_1d[j-1]))
            atr_sum += tr
        atr = atr_sum / 14
        max_high = np.max(high_1d[i-13:i+1].values)
        min_low = np.min(low_1d[i-13:i+1].values)
        if max_high != min_low:
            chop_1d[i] = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
        else:
            chop_1d[i] = 50  # neutral when no range
    
    # Align 1d data to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate volume ratio (current vs 20-period average)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        vol_r = vol_ratio[i]
        ch = chop_aligned[i]
        price = close[i]
        
        if np.isnan(vol_r) or np.isnan(ch):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        h4 = camarilla_h4_aligned[i]
        l4 = camarilla_l4_aligned[i]
        
        if np.isnan(r3) or np.isnan(s3) or np.isnan(h4) or np.isnan(l4):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            if price > h4 or ch > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if price < l4 or ch > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if price <= s3 and vol_r > 2.0 and ch < 61.8:
                position = -1
                signals[i] = -0.25
            elif price >= r3 and vol_r > 2.0 and ch < 61.8:
                position = 1
                signals[i] = 0.25
    
    return signals