#!/usr/bin/env python3
"""
1d_camarilla_weekly_trend_v1
Hypothesis: Uses daily Camarilla levels with weekly trend filter for 1d timeframe.
- Long when price crosses above daily R3 with weekly bullish trend
- Short when price crosses below daily S3 with weekly bearish trend
- Exits when price returns to daily pivot or weekly trend reverses
- Targets 10-25 trades/year to minimize fee decay
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_weekly_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get daily data (same as input since timeframe is 1d)
    df_1d = prices.copy()
    
    # Calculate Camarilla levels for previous day
    # Using previous day's OHLC to avoid look-ahead
    high_1d_prev = df_1d['high'].shift(1).values
    low_1d_prev = df_1d['low'].shift(1).values
    close_1d_prev = df_1d['close'].shift(1).values
    
    # Camarilla levels (based on previous day's range)
    R3 = close_1d_prev + 1.1 * (high_1d_prev - low_1d_prev) * 1.1 / 4
    S3 = close_1d_prev - 1.1 * (high_1d_prev - low_1d_prev) * 1.1 / 4
    pivot = (high_1d_prev + low_1d_prev + close_1d_prev) / 3
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly pivot point: (H+L+C)/3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = (high_1w + low_1w + close_1w) / 3
    
    # Weekly trend: bullish if close > pivot, bearish if close < pivot
    weekly_bullish = close_1w > weekly_pivot
    weekly_bearish = close_1w < weekly_pivot
    
    # Align weekly trend to daily timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 1  # Need at least one previous day
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or np.isnan(pivot[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price returns to pivot or weekly turns bearish
            if close[i] <= pivot[i] or weekly_bearish_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to pivot or weekly turns bullish
            if close[i] >= pivot[i] or weekly_bullish_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price crosses above R3 with weekly bullish trend
            if (close[i] > R3[i] and close[i-1] <= R3[i] and 
                weekly_bullish_aligned[i] > 0.5):
                position = 1
                signals[i] = 0.25
            # Short entry: Price crosses below S3 with weekly bearish trend
            elif (close[i] < S3[i] and close[i-1] >= S3[i] and 
                  weekly_bearish_aligned[i] > 0.5):
                position = -1
                signals[i] = -0.25
    
    return signals