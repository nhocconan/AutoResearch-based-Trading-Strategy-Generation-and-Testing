#!/usr/bin/env python3
"""
6h_Turtle_Soup_Reversal
Hypothesis: Turtle Soup strategy exploits false breakouts at daily highs/lows.
In strong trends, price often tests and reverses at key levels. Uses 1d high/low
as traps, with 1w trend filter to avoid counter-trend traps. Designed for low
frequency (15-30 trades/year) to minimize fee drag in ranging/choppy markets.
Works in both bull/bear by fading false breaks in direction of higher timeframe trend.
"""

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
    
    # Get daily data for traps
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly 20-period EMA for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Previous day's high and low as traps
    prev_high = df_1d['high'].shift(1).values  # Previous day's high
    prev_low = df_1d['low'].shift(1).values    # Previous day's low
    
    # Align traps to 6h timeframe (from previous day's close)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for weekly EMA to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(prev_high_aligned[i]) or
            np.isnan(prev_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Turtle Soup: fade false breakouts of previous day's high/low
        # Long setup: price tests and fails to hold above previous day's high
        long_setup = (high[i] > prev_high_aligned[i]) and (close[i] < prev_high_aligned[i])
        # Short setup: price tests and fails to hold below previous day's low
        short_setup = (low[i] < prev_low_aligned[i]) and (close[i] > prev_low_aligned[i])
        
        # Only take trades in direction of weekly trend
        long_entry = long_setup and uptrend
        short_entry = short_setup and downtrend
        
        # Exit: reverse signal or price reaches opposite trap level
        long_exit = (position == 1) and (low[i] <= prev_low_aligned[i] or short_setup)
        short_exit = (position == -1) and (high[i] >= prev_high_aligned[i] or long_setup)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Turtle_Soup_Reversal"
timeframe = "6h"
leverage = 1.0