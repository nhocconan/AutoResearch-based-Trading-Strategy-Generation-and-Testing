#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot filter and volume confirmation.
Long when price breaks above Donchian(20) high AND price > weekly R1 pivot AND volume > 1.5x 20-period average.
Short when price breaks below Donchian(20) low AND price < weekly S1 pivot AND volume > 1.5x 20-period average.
Exit when price crosses Donchian(20) midpoint or volume drops below average.
Uses 1d for weekly pivots (calculated from prior week), 6h for Donchian and volume.
Target: 50-150 total trades over 4 years (12-37/year).
"""

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
    
    # Get 1d data for weekly pivots (prior week's high/low/close)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using prior week's data)
    # Weekly R1 = 2 * PP - Low_week, Weekly S1 = 2 * PP - High_week
    # PP = (High_week + Low_week + Close_week) / 3
    # We need to shift by 1 week to avoid look-ahead (use prior week's data)
    weekly_high = pd.Series(high_1d).shift(5).values  # Approximate prior week (5 trading days)
    weekly_low = pd.Series(low_1d).shift(5).values
    weekly_close = pd.Series(close_1d).shift(5).values
    
    # Calculate pivot points
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    
    # Align weekly pivots to 6h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 6h Donchian(20) - use shifted to avoid look-ahead
    donchian_high = pd.Series(high).shift(1).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).shift(1).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # 6h volume confirmation - 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long conditions: break above Donchian high, above weekly R1, volume > 1.5x average
            if (price > donchian_high[i] and 
                price > r1_aligned[i] and 
                vol > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian low, below weekly S1, volume > 1.5x average
            elif (price < donchian_low[i] and 
                  price < s1_aligned[i] and 
                  vol > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint OR volume drops below average
            if price < donchian_mid[i] or vol < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint OR volume drops below average
            if price > donchian_mid[i] or vol < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0