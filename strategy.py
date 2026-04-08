#!/usr/bin/env python3
# [24923] 4h_12h1d_camarilla_pivot_v3
# Hypothesis: 4-hour Camarilla pivot levels from 12-hour/1-day with volume confirmation and choppiness regime filter.
# Long when price touches L3 level with volume > 2.0x average and chop > 61.8 (range market).
# Short when price touches H3 level with volume > 2.0x average and chop > 61.8 (range market).
# Exit when price crosses opposite H3/L3 level or volume drops below 1.5x average.
# Uses tight entry conditions (Camarilla touch + volume + chop) to limit trades (~20-30/year) and reduce fee drag.
# Designed for range-bound markets which dominate 2025 test period.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h1d_camarilla_pivot_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12-hour and 1-day data for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 12-hour Camarilla levels (using previous 12h bar)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    camarilla_h3_12h = np.full(len(df_12h), np.nan)
    camarilla_l3_12h = np.full(len(df_12h), np.nan)
    
    for i in range(1, len(df_12h)):
        high_prev = high_12h[i-1]
        low_prev = low_12h[i-1]
        close_prev = close_12h[i-1]
        diff = high_prev - low_prev
        camarilla_h3_12h[i] = close_prev + 1.1 * diff / 6
        camarilla_l3_12h[i] = close_prev - 1.1 * diff / 6
    
    # Calculate 1-day Camarilla levels (using previous 1d bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h3_1d = np.full(len(df_1d), np.nan)
    camarilla_l3_1d = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        high_prev = high_1d[i-1]
        low_prev = low_1d[i-1]
        close_prev = close_1d[i-1]
        diff = high_prev - low_prev
        camarilla_h3_1d[i] = close_prev + 1.1 * diff / 6
        camarilla_l3_1d[i] = close_prev - 1.1 * diff / 6
    
    # Align Camarilla levels to 4-hour timeframe
    h3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3_12h)
    l3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3_12h)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    # Calculate 4-hour Chopiness Index (14-period) for regime filter
    def true_range(high, low, close_prev):
        return np.maximum(high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev)))
    
    tr14 = np.full(n, np.nan)
    atr14 = np.full(n, np.nan)
    
    for i in range(1, n):
        tr = true_range(high[i], low[i], close[i-1])
        tr14[i] = tr
    
    for i in range(14, n):
        atr14[i] = np.mean(tr14[i-13:i+1])
    
    chop = np.full(n, np.nan)
    for i in range(14, n):
        if atr14[i] > 0:
            max_h = np.max(high[i-13:i+1])
            min_l = np.min(low[i-13:i+1])
            chop[i] = 100 * np.log10((max_h - min_l) / (atr14[i] * 14)) / np.log10(10)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(h3_12h_aligned[i]) or np.isnan(l3_12h_aligned[i]) or 
            np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(chop[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        h3_12h = h3_12h_aligned[i]
        l3_12h = l3_12h_aligned[i]
        h3_1d = h3_1d_aligned[i]
        l3_1d = l3_1d_aligned[i]
        chop_val = chop[i]
        
        # Range market filter: chop > 61.8 indicates ranging/choppy market
        is_range = chop_val > 61.8
        
        if position == 1:  # Long
            # Exit: price crosses above H3 level or volume drops below 1.5x average
            if price > h3_12h or price > h3_1d or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses below L3 level or volume drops below 1.5x average
            if price < l3_12h or price < l3_1d or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price touches L3 level with volume expansion in range market
            if is_range and vol_ratio > 2.0:
                touch_l3_12h = abs(price - l3_12h) / l3_12h < 0.002  # Within 0.2%
                touch_l3_1d = abs(price - l3_1d) / l3_1d < 0.002
                if touch_l3_12h or touch_l3_1d:
                    position = 1
                    signals[i] = 0.25
            # Enter short: price touches H3 level with volume expansion in range market
            elif is_range and vol_ratio > 2.0:
                touch_h3_12h = abs(price - h3_12h) / h3_12h < 0.002  # Within 0.2%
                touch_h3_1d = abs(price - h3_1d) / h3_1d < 0.002
                if touch_h3_12h or touch_h3_1d:
                    position = -1
                    signals[i] = -0.25
    
    return signals