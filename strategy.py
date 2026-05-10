#!/usr/bin/env python3
"""
6h_TurtleSoup_Reverse_Entry
Hypothesis: Turtle Soup strategy (false breakout reversal) on 6h timeframe. 
Enter when price briefly breaks Donchian(20) high/low but immediately reverses back inside the channel,
indicating a failed breakout and potential reversal. Use 1w trend filter to align with higher timeframe direction.
Works in both bull/bear markets by fading false breakouts. Target: 60-120 total trades over 4 years (15-30/year).
"""

name = "6h_TurtleSoup_Reverse_Entry"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Get price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian (20) + weekly EMA (50)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Turtle Soup Long: price breaks below 20-period low but closes back above it
            # Only in weekly uptrend
            if low[i] < lowest_low[i] and close[i] > lowest_low[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Turtle Soup Short: price breaks above 20-period high but closes back below it
            # Only in weekly downtrend
            elif high[i] > highest_high[i] and close[i] < highest_high[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low OR weekly trend turns bearish
            if low[i] < lowest_low[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high OR weekly trend turns bullish
            if high[i] > highest_high[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals