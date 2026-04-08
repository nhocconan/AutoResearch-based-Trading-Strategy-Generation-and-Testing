#!/usr/bin/env python3
"""
4h_camarilla_1d_trend_v1
Hypothesis: Uses 1-day Camarilla levels with 4-hour price action for entries.
- Long when price touches 4h low near daily S3 with bullish 1d trend
- Short when price touches 4h high near daily R3 with bearish 1d trend
- Exits when price returns to daily pivot or trend reverses
- Targets 20-40 trades/year to minimize fee decay
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_1d_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for previous day (to avoid look-ahead)
    high_1d_prev = df_1d['high'].shift(1).values
    low_1d_prev = df_1d['low'].shift(1).values
    close_1d_prev = df_1d['close'].shift(1).values
    
    # Camarilla levels (based on previous day's range)
    R3 = close_1d_prev + 1.1 * (high_1d_prev - low_1d_prev) * 1.1 / 4
    S3 = close_1d_prev - 1.1 * (high_1d_prev - low_1d_prev) * 1.1 / 4
    pivot = (high_1d_prev + low_1d_prev + close_1d_prev) / 3
    
    # Daily trend: bullish if close > pivot, bearish if close < pivot
    daily_bullish = df_1d['close'] > df_1d['pivot'] if 'pivot' in df_1d else df_1d['close'] > (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    daily_bearish = df_1d['close'] < df_1d['pivot'] if 'pivot' in df_1d else df_1d['close'] < (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    
    # Calculate daily pivot for trend (using previous day's data)
    daily_pivot = (high_1d_prev + low_1d_prev + close_1d_prev) / 3
    daily_bullish = close_1d_prev > daily_pivot
    daily_bearish = close_1d_prev < daily_pivot
    
    # Align daily data to 4h timeframe
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
    daily_bullish_4h = align_htf_to_ltf(prices, df_1d, daily_bullish.astype(float))
    daily_bearish_4h = align_htf_to_ltf(prices, df_1d, daily_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 1  # Need at least one previous day
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R3_4h[i]) or np.isnan(S3_4h[i]) or np.isnan(pivot_4h[i]) or
            np.isnan(daily_bullish_4h[i]) or np.isnan(daily_bearish_4h[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price returns to pivot or daily turns bearish
            if low[i] <= pivot_4h[i] or daily_bearish_4h[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to pivot or daily turns bullish
            if high[i] >= pivot_4h[i] or daily_bullish_4h[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price touches 4h low near daily S3 with daily bullish trend
            # Allow small tolerance (0.1%) for touch
            low_touch_s3 = low[i] <= S3_4h[i] * 1.001
            if (low_touch_s3 and daily_bullish_4h[i] > 0.5):
                position = 1
                signals[i] = 0.25
            # Short entry: Price touches 4h high near daily R3 with daily bearish trend
            elif high[i] >= R3_4h[i] * 0.999 and daily_bearish_4h[i] > 0.5:
                position = -1
                signals[i] = -0.25
    
    return signals