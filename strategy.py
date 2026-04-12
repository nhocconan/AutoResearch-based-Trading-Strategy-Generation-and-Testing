#!/usr/bin/env python3
"""
6h_1d_Weekly_Pivot_Donchian_Breakout
Hypothesis: 6s timeframe with weekly pivot levels (from Monday open) and daily Donchian breakout.
Weekly pivots provide structural support/resistance that holds across market regimes.
Donchian(20) breakout on daily timeframe filters for momentum in direction of weekly bias.
Only takes longs when price above weekly pivot and breaks daily Donchian high,
only shorts when below weekly pivot and breaks daily Donchian low.
Designed for low turnover (target 15-35 trades/year) by requiring confluence of
weekly structure and daily momentum breakout.
Works in bull/bear markets by using weekly pivot as dynamic bias filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Weekly_Pivot_Donchian_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load daily data ONCE before loop for Donchian and weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot from Monday open of current week
    # We'll approximate: weekly pivot = (weekly high + weekly low + weekly close) / 3
    # For simplicity, use prior week's values to avoid look-ahead
    weekly_high = df_1d['high'].rolling(window=5, min_periods=5).max().shift(5).values
    weekly_low = df_1d['low'].rolling(window=5, min_periods=5).min().shift(5).values
    weekly_close = df_1d['close'].shift(5).values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    
    # Calculate daily Donchian channels (20-period)
    donchian_high = df_1d['high'].rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = df_1d['low'].rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align weekly pivot and daily Donchian to 6h timeframe
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    donchian_high_6h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_6h = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(weekly_pivot_6h[i]) or np.isnan(donchian_high_6h[i]) or 
            np.isnan(donchian_low_6h[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions: Donchian breakout in direction of weekly pivot bias
        long_entry = (close[i] > donchian_high_6h[i]) and (close[i] > weekly_pivot_6h[i])
        short_entry = (close[i] < donchian_low_6h[i]) and (close[i] < weekly_pivot_6h[i])
        
        # Exit conditions: return to opposite Donchian level or cross weekly pivot
        long_exit = (close[i] < donchian_low_6h[i]) or (close[i] < weekly_pivot_6h[i])
        short_exit = (close[i] > donchian_high_6h[i]) or (close[i] > weekly_pivot_6h[i])
        
        # Priority: entry > exit > hold
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