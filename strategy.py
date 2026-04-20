#!/usr/bin/env python3
"""
1d_Donchian_20_With_Weekly_Trend_Filter
Hypothesis: Trade Donchian(20) breakouts on 1d with weekly trend filter. 
Long when price breaks above 20-day high + weekly uptrend.
Short when price breaks below 20-day low + weekly downtrend.
Donchian provides clear breakout signals; weekly trend filter ensures alignment with higher timeframe momentum.
Target: 30-100 total trades over 4 years (7-25/year) with position size 0.25.
Works in bull/bear: weekly filter avoids counter-trend trades, Donchian captures strong moves.
"""

name = "1d_Donchian_20_With_Weekly_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_weekly = df_weekly['close'].values
    ema20_weekly = np.full_like(close_weekly, np.nan)
    if len(close_weekly) >= 20:
        multiplier = 2.0 / (20 + 1)
        ema20_weekly[19] = np.mean(close_weekly[:20])
        for i in range(20, len(close_weekly)):
            ema20_weekly[i] = multiplier * close_weekly[i] + (1 - multiplier) * ema20_weekly[i-1]
    ema20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    
    # Calculate Donchian channels (20-day high/low)
    high_20 = np.full(n, np.nan)
    low_20 = np.full(n, np.nan)
    
    for i in range(19, n):
        high_20[i] = np.max(high[i-19:i+1])
        low_20[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure Donchian is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema20_weekly_aligned[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 20-day high + weekly uptrend
            if close[i] > high_20[i] and close[i] > ema20_weekly_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low + weekly downtrend
            elif close[i] < low_20[i] and close[i] < ema20_weekly_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 20-day low OR weekly trend turns down
            if close[i] < low_20[i] or close[i] < ema20_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 20-day high OR weekly trend turns up
            if close[i] > high_20[i] or close[i] > ema20_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals