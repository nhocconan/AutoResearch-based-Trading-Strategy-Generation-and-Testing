#!/usr/bin/env python3
# 6h_Donchian_20_WeeklyTrend_Pullback
# Hypothesis: Use weekly trend filter (price above/below 200 EMA) with Donchian(20) breakout and pullback entry on 6h.
# In bull markets, buy pullbacks to 20-period EMA after upward breakout; in bear markets, sell rallies to 20-period EMA after downward breakout.
# Weekly trend filter ensures we only trade in the dominant long-term direction, reducing whipsaws.
# Donchian breakouts capture momentum, and pullback entries improve risk-reward.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Donchian_20_WeeklyTrend_Pullback"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter (200 EMA)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Get daily data for Donchian(20) channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels (20-period high/low)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Get 6h data for 20-period EMA (pullback filter)
    ema20_6h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).values
    
    # Align weekly EMA200 to 6h
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Align daily Donchian channels to 6h
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema200_1w_aligned[i]) or np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or np.isnan(ema20_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high + above weekly EMA200 + pullback to EMA20
            if close[i] > high_20_aligned[i] and close[i] > ema200_1w_aligned[i] and close[i] <= ema20_6h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low + below weekly EMA200 + pullback to EMA20
            elif close[i] < low_20_aligned[i] and close[i] < ema200_1w_aligned[i] and close[i] >= ema20_6h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low or below weekly EMA200
            if close[i] < low_20_aligned[i] or close[i] < ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high or above weekly EMA200
            if close[i] > high_20_aligned[i] or close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals