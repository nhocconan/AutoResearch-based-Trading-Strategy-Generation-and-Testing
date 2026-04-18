#!/usr/bin/env python3
"""
12h_Pivot_R1S1_Breakout_Volume_Confirmation_v2
Hypothesis: Use daily Camarilla pivot levels R1/S1 as support/resistance on 12h timeframe. Enter long when price breaks above R1 with volume confirmation, short when price breaks below S1 with volume confirmation. Exit at opposite pivot level (S1 for long, R1 for short) or on reversal. Uses tight entry conditions to limit trades to 12-37/year, reducing fee drag. Designed to work in both bull and bear markets by trading breakouts from key daily levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period."""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close
    c = close
    h = high
    l = low
    r1 = c + (range_val * 1.1 / 12)
    s1 = c - (range_val * 1.1 / 12)
    r2 = c + (range_val * 1.1 / 6)
    s2 = c - (range_val * 1.1 / 6)
    return r1, s1, r2, s2

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels
    r1_1d = np.zeros(len(close_1d))
    s1_1d = np.zeros(len(close_1d))
    r2_1d = np.zeros(len(close_1d))
    s2_1d = np.zeros(len(close_1d))
    
    for i in range(len(close_1d)):
        r1, s1, r2, s2 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        r1_1d[i] = r1
        s1_1d[i] = s1
        r2_1d[i] = r2
        s2_1d[i] = s2
    
    # Align daily pivots to 12h timeframe (wait for daily close)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Volume confirmation: >2.0x 24-period average (24*12h = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Need volume MA history
    
    for i in range(start_idx, n):
        if (np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        r2 = r2_1d_aligned[i]
        s2 = s2_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: break above R1 with volume
            if price > r1 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume
            elif price < s1 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price reaches S1 (opposite level) or reverses below R1
            if price <= s1 or price < r1:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price reaches R1 (opposite level) or reverses above S1
            if price >= r1 or price > s1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Pivot_R1S1_Breakout_Volume_Confirmation_v2"
timeframe = "12h"
leverage = 1.0