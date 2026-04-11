#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d pivot-based reversal and volume confirmation.
Uses Woodie's pivot points from daily data to identify potential reversal zones.
Enters long near S1/S2 support with bullish volume divergence, short near R1/R2 resistance with bearish volume divergence.
Designed to work in both bull (buy dips) and bear (sell rallies) markets by fading extremes.
Target: 20-40 trades/year (~80-160 total over 4 years) with disciplined risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_woodie_pivot_v1"
timeframe = "6h"
leverage = 1.0

def calculate_woodie_pivot(high, low, close):
    """Calculate Woodie's pivot points: P = (H+L+2C)/4, R1 = 2P-L, S1 = 2P-H, etc."""
    pivot = (high + low + 2 * close) / 4
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, s1, r2, s2

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Woodie's pivot points for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot, r1, s1, r2, s2 = calculate_woodie_pivot(high_1d, low_1d, close_1d)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 20 to ensure volume MA is valid
    for i in range(20, n):
        # Skip if pivot data is invalid
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Price position relative to pivot levels
        price = close[i]
        near_support = (price <= s1_aligned[i] * 1.005) and (price >= s2_aligned[i] * 0.995)
        near_resistance = (price >= r1_aligned[i] * 0.995) and (price <= r2_aligned[i] * 1.005)
        
        # Entry conditions
        long_entry = near_support and vol_confirm
        short_entry = near_resistance and vol_confirm
        
        # Exit conditions: return to pivot or opposite extreme
        long_exit = price >= pivot_aligned[i] or price <= s2_aligned[i] * 0.99
        short_exit = price <= pivot_aligned[i] or price >= r2_aligned[i] * 1.01
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals