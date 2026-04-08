#!/usr/bin/env python3
# [24991] 6h_1d_1w_camarilla_pivot_fade_breakout_v1
# Hypothesis: 6-hour Camarilla pivot-based strategy using daily and weekly pivots.
# Long when price breaks above daily R4 with volume > 1.5x average and price > weekly pivot.
# Short when price breaks below daily S4 with volume > 1.5x average and price < weekly pivot.
# Exit when price returns to daily pivot.
# Uses Camarilla levels for breakout/fade signals with weekly trend filter.
# Designed to generate ~15-30 trades/year to avoid fee dust while capturing momentum.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_camarilla_pivot_fade_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    range_1d = high_1d - low_1d
    close_prev = np.roll(close_1d, 1)
    close_prev[0] = close_1d[0]  # avoid NaN on first element
    pivot_1d = (high_1d + low_1d + close_prev) / 3.0
    r4_1d = pivot_1d + (high_1d - low_1d) * 1.5
    s4_1d = pivot_1d - (high_1d - low_1d) * 1.5
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    close_prev_1w = np.roll(close_1w, 1)
    close_prev_1w[0] = close_1w[0]
    pivot_1w = (high_1w + low_1w + close_prev_1w) / 3.0
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align daily and weekly pivots to 6-hour timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(pivot_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or 
            np.isnan(pivot_1w_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        
        if position == 1:  # Long
            # Exit: price returns to daily pivot
            if price <= pivot_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to daily pivot
            if price >= pivot_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above daily R4 with volume expansion and above weekly pivot
            if price > r4_1d_aligned[i] and vol_ratio > 1.5 and price > pivot_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below daily S4 with volume expansion and below weekly pivot
            elif price < s4_1d_aligned[i] and vol_ratio > 1.5 and price < pivot_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals