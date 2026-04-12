#!/usr/bin/env python3
"""
4h_1d_Camarilla_Breakout_Volume_Confirmation_v1
Hypothesis: Use daily Camarilla pivot levels with volume spike confirmation on 4h timeframe.
Buy when price breaks above H4 level with volume > 1.5x 20-period average.
Sell when price breaks below L4 level with volume > 1.5x 20-period average.
Works in both bull/bear by capturing breakouts with volume confirmation, avoiding false breakouts.
Uses tight entry conditions to limit trades (<400 total) and reduce fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Breakout_Volume_Confirmation_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].iloc[-2] if len(df_1d) >= 2 else df_1d['high'].iloc[-1]
    prev_low = df_1d['low'].iloc[-2] if len(df_1d) >= 2 else df_1d['low'].iloc[-1]
    prev_close = df_1d['close'].iloc[-2] if len(df_1d) >= 2 else df_1d['close'].iloc[-1]
    
    # Camarilla levels (based on previous day)
    range_val = prev_high - prev_low
    h4 = prev_close + 1.1 * range_val * 1.1 / 2
    l4 = prev_close - 1.1 * range_val * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (same value throughout the day)
    h4_array = np.full(len(df_1d), h4)
    l4_array = np.full(len(df_1d), l4)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4_array)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4_array)
    
    # Volume confirmation: 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any data invalid
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        # Breakout conditions
        long_breakout = close[i] > h4_aligned[i] and volume_spike
        short_breakout = close[i] < l4_aligned[i] and volume_spike
        
        # Exit conditions: return to mid-point (pivot)
        pivot = (prev_high + prev_low + prev_close) / 3
        pivot_array = np.full(len(df_1d), pivot)
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_array)
        
        long_exit = close[i] < pivot_aligned[i]
        short_exit = close[i] > pivot_aligned[i]
        
        # Signal logic
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals