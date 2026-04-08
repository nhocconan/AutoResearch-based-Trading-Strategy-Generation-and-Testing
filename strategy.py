#!/usr/bin/env python3
# 12h_1d_1w_camarilla_volume_chop_v1
# Hypothesis: 12h Camarilla pivot reversals with 1d volume confirmation and 1w chop regime filter.
# Long: price touches Camarilla L3 support with volume > 1.3x average AND weekly chop < 61.8 (trending)
# Short: price touches Camarilla H3 resistance with volume > 1.3x average AND weekly chop < 61.8 (trending)
# Exit: price reaches opposite Camarilla level (H3 for long, L3 for short) or chop > 61.8 (ranging)
# Uses 12h primary timeframe with 1d/1w HTF for confirmation to reduce overtrading and improve Sharpe.
# Designed for low trade frequency (target: 50-150 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_camarilla_volume_chop_v1"
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
    
    # Calculate 12h Camarilla levels (based on previous bar's range)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    
    for i in range(1, n):
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        range_val = prev_high - prev_low
        
        if range_val > 0:
            camarilla_h3[i] = prev_close + range_val * 1.1 / 4
            camarilla_l3[i] = prev_close - range_val * 1.1 / 4
            camarilla_h4[i] = prev_close + range_val * 1.1 / 2
            camarilla_l4[i] = prev_close - range_val * 1.1 / 2
    
    # Calculate volume ratio (current vs 20-period average)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 0)
    
    # Get 1d data for additional volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume ratio
    vol_1d = df_1d['volume'].values
    vol_sma_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_sma_1d[i] = np.mean(vol_1d[i-20:i])
    vol_ratio_1d = np.where(vol_sma_1d > 0, vol_1d / vol_sma_1d, 0)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # Get 1w data for chop regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate Chopiness Index on 1w data (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    chop_1w = np.full(len(df_1w), np.nan)
    for i in range(14, len(df_1w)):
        atr_sum = 0
        for j in range(i-13, i+1):
            tr = max(high_1w[j] - low_1w[j],
                     abs(high_1w[j] - close_1w[j-1]),
                     abs(low_1w[j] - close_1w[j-1]))
            atr_sum += tr
        atr = atr_sum / 14
        max_high = np.max(high_1w[i-13:i+1])
        min_low = np.min(low_1w[i-13:i+1])
        if max_high != min_low:
            chop_1w[i] = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
        else:
            chop_1w[i] = 50  # neutral when no range
    
    # Align 1w chop to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        vol_r = vol_ratio[i]
        vol_r_1d = vol_ratio_1d_aligned[i]
        ch = chop_aligned[i]
        price = close[i]
        
        if np.isnan(vol_r) or np.isnan(vol_r_1d) or np.isnan(ch):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        h3 = camarilla_h3[i]
        l3 = camarilla_l3[i]
        h4 = camarilla_h4[i]
        l4 = camarilla_l4[i]
        
        if np.isnan(h3) or np.isnan(l3):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            if price >= h3 or ch > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if price <= l3 or ch > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: price touches L3 support with volume confirmation in trending market
            if price <= l3 and vol_r > 1.3 and vol_r_1d > 1.2 and ch < 61.8:
                position = 1
                signals[i] = 0.25
            # Short entry: price touches H3 resistance with volume confirmation in trending market
            elif price >= h3 and vol_r > 1.3 and vol_r_1d > 1.2 and ch < 61.8:
                position = -1
                signals[i] = -0.25
    
    return signals