#!/usr/bin/env python3
"""
6h_1w1d_camarilla_pivot_v1
Hypothesis: 6-hour strategy using weekly and daily context with Camarilla pivot levels.
Long when price breaks above weekly R4 with daily close above weekly Pivot and volume > 2x average.
Short when price breaks below weekly S4 with daily close below weekly Pivot and volume > 2x average.
Exit when price returns to weekly Pivot or volume drops below 1.5x average.
Uses discrete position sizing (0.25) to reduce trade frequency. Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w1d_camarilla_pivot_v1"
timeframe = "6h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    if len(high) < 1:
        return np.full(len(high), np.nan), np.full(len(high), np.nan)
    
    pivot = (high + low + close) / 3.0
    range_val = high - low
    
    # Standard Camarilla multipliers
    H3 = pivot + (range_val * 1.1 / 4)
    L3 = pivot - (range_val * 1.1 / 4)
    H4 = pivot + (range_val * 1.1 / 2)
    L4 = pivot - (range_val * 1.1 / 2)
    
    return H3, L3, H4, L4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for context and Pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly Pivot (using previous week's data)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    # Weekly support/resistance levels (Standard Camarilla S4/R4)
    S4_1w = pivot_1w - (range_1w * 1.1 / 2)
    R4_1w = pivot_1w + (range_1w * 1.1 / 2)
    
    # Calculate daily close for trend filter (using previous day's data)
    close_1d = df_1d['close'].values
    
    # Align indicators to 6-hour timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    S4_1w_aligned = align_htf_to_ltf(prices, df_1w, S4_1w)
    R4_1w_aligned = align_htf_to_ltf(prices, df_1w, R4_1w)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(S4_1w_aligned[i]) or 
            np.isnan(R4_1w_aligned[i]) or np.isnan(close_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        pivot = pivot_1w_aligned[i]
        S4 = S4_1w_aligned[i]
        R4 = R4_1w_aligned[i]
        daily_close = close_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price returns to weekly Pivot or volume drops below 1.5x average
            if price <= pivot or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to weekly Pivot or volume drops below 1.5x average
            if price >= pivot or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above weekly R4 with daily close above weekly Pivot and volume expansion
            if price > R4 and daily_close > pivot and vol_ratio > 2.0:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below weekly S4 with daily close below weekly Pivot and volume expansion
            elif price < S4 and daily_close < pivot and vol_ratio > 2.0:
                position = -1
                signals[i] = -0.25
    
    return signals