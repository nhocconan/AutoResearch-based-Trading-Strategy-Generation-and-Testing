#!/usr/bin/env python3
"""
12h_weekly_trend_with_daily_pullback_v1
Hypothesis: Weekly trend filter with daily pullback to value area for continuation.
- Only trade in direction of weekly trend (above/below weekly pivot)
- Long: Weekly bullish + price pulls back to daily pivot then closes above it
- Short: Weekly bearish + price pulls back to daily pivot then closes below it
- Exit on opposite pullback or weekly trend reversal
- Target: 15-25 trades/year to avoid overtrading
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_weekly_trend_with_daily_pullback_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Daily data for pivot calculation (12h data is used as proxy for daily)
    # Since we're on 12h timeframe, we'll use 12h bars for pivot (more frequent)
    high_12h = high
    low_12h = low
    close_12h = close
    
    # Previous 12h bar's OHLC for pivot
    high_12h_prev = np.roll(high_12h, 1)
    low_12h_prev = np.roll(low_12h, 1)
    close_12h_prev = np.roll(close_12h, 1)
    # Set first value to NaN to avoid using uninitialized data
    high_12h_prev[0] = np.nan
    low_12h_prev[0] = np.nan
    close_12h_prev[0] = np.nan
    
    # Pivot point (value area)
    pivot = (high_12h_prev + low_12h_prev + close_12h_prev) / 3
    
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
    
    # Align weekly trend to 12h
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