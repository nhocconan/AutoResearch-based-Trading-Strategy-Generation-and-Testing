#!/usr/bin/env python3
"""
6h_1d_WeeklyPivot_DonchianBreakout_v2
Hypothesis: Combine weekly pivot points with 6-hour Donchian channel breakouts and volume confirmation.
Weekly pivot provides directional bias from higher timeframe, while Donchian breakout captures momentum.
Volume filter ensures breakouts have conviction. Designed to work in both bull and bear markets by
focusing on institutional levels and breakout confirmation. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_WeeklyPivot_DonchianBreakout_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY PIVOT CALCULATION ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (P), resistance (R1,R2), support (S1,S2)
    pivot = np.zeros(len(df_1w))
    R1 = np.zeros(len(df_1w))
    S1 = np.zeros(len(df_1w))
    R2 = np.zeros(len(df_1w))
    S2 = np.zeros(len(df_1w))
    
    for i in range(len(df_1w)):
        pp = (high_1w[i] + low_1w[i] + close_1w[i]) / 3
        r = high_1w[i] - low_1w[i]
        pivot[i] = pp
        R1[i] = 2 * pp - low_1w[i]
        S1[i] = 2 * pp - high_1w[i]
        R2[i] = pp + r
        S2[i] = pp - r
    
    # Align weekly pivot levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot)
    R1_6h = align_htf_to_ltf(prices, df_1w, R1)
    S1_6h = align_htf_to_ltf(prices, df_1w, S1)
    R2_6h = align_htf_to_ltf(prices, df_1w, R2)
    S2_6h = align_htf_to_ltf(prices, df_1w, S2)
    
    # === DAILY DONCHIAN CHANNEL (20-period) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 6h timeframe
    upper_6h = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_6h = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # === VOLUME FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(pivot_6h[i]) or np.isnan(R1_6h[i]) or np.isnan(S1_6h[i]) or
            np.isnan(upper_6h[i]) or np.isnan(lower_6h[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine bias from weekly pivot
        bullish_bias = close[i] > pivot_6h[i]
        bearish_bias = close[i] < pivot_6h[i]
        
        # Breakout conditions with volume confirmation
        long_breakout = (close[i] > upper_6h[i]) and (vol_ratio[i] > 1.5) and bullish_bias
        short_breakout = (close[i] < lower_6h[i]) and (vol_ratio[i] > 1.5) and bearish_bias
        
        # Exit conditions: return to opposite pivot level
        exit_long = (position == 1) and (close[i] < S1_6h[i])
        exit_short = (position == -1) and (close[i] > R1_6h[i])
        
        # Execute trades
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals