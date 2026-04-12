#!/usr/bin/env python3
"""
12h_1d_camarilla_pivot_volume
Uses 12h price with 1d Camarilla pivot levels, volume confirmation, and trend filter.
Long when price > H4 and above EMA200, short when price < L4 and below EMA200.
Exit when price returns to Pivot or reverses at H3/L3.
Designed for low trade frequency (target: 12-30 trades/year) to minimize fee drift.
Works in both trending and ranging markets by combining pivot levels with trend filter.
"""

name = "12h_1d_camarilla_pivot_volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla pivot levels
    # Pivot = (High + Low + Close) / 3
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # H3 = Close + 1.25 * (High - Low)
    # L3 = Close - 1.25 * (High - Low)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    h4 = close_1d + 1.5 * range_hl
    l4 = close_1d - 1.5 * range_hl
    h3 = close_1d + 1.25 * range_hl
    l3 = close_1d - 1.25 * range_hl
    
    # Align Camarilla levels to 12h (using previous day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Trend filter: EMA200 on 12h
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(ema200[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price > H4 and above EMA200, with volume confirmation
        if close[i] > h4_aligned[i] and close[i] > ema200[i] and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short entry: price < L4 and below EMA200, with volume confirmation
        elif close[i] < l4_aligned[i] and close[i] < ema200[i] and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions
        elif position == 1 and (close[i] <= pivot_aligned[i] or close[i] >= h3_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] >= pivot_aligned[i] or close[i] <= l3_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals