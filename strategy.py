#!/usr/bin/env python3
"""
1d_camarilla_pullback_v1
Hypothesis: Weekly trend + daily pullback to value area (Pivot) for continuation.
- Only trade in direction of weekly trend (above/below weekly pivot)
- Long: Weekly bullish + price pulls back to daily pivot then closes above it
- Short: Weekly bearish + price pulls back to daily pivot then closes below it
- Exit on opposite pullback or weekly trend reversal
- Target: 15-25 trades/year to avoid overtrading
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pullback_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Daily data (same as input)
    df_1d = prices.copy()
    
    # Previous day's OHLC for daily pivot
    high_1d_prev = df_1d['high'].shift(1).values
    low_1d_prev = df_1d['low'].shift(1).values
    close_1d_prev = df_1d['close'].shift(1).values
    
    # Daily pivot point (value area)
    pivot = (high_1d_prev + low_1d_prev + close_1d_prev) / 3
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = (high_1w + low_1w + close_1w) / 3
    weekly_bullish = close_1w > weekly_pivot
    weekly_bearish = close_1w < weekly_pivot
    
    # Align weekly trend to daily
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(pivot[i]) or np.isnan(weekly_bullish_aligned[i]) or 
            np.isnan(weekly_bearish_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long
            # Exit: pullback below pivot or weekly turns bearish
            if close[i] < pivot[i] or weekly_bearish_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: pullback above pivot or weekly turns bullish
            if close[i] > pivot[i] or weekly_bullish_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long: weekly bullish + pullback to pivot then close above
            if (weekly_bullish_aligned[i] > 0.5 and 
                close[i-1] <= pivot[i-1] and close[i] > pivot[i]):
                position = 1
                signals[i] = 0.25
            # Short: weekly bearish + pullback to pivot then close below
            elif (weekly_bearish_aligned[i] > 0.5 and 
                  close[i-1] >= pivot[i-1] and close[i] < pivot[i]):
                position = -1
                signals[i] = -0.25
    
    return signals