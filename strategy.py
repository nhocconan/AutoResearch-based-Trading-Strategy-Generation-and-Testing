#!/usr/bin/env python3
# Hypothesis: 6h Donchian breakout with weekly pivot bias and volume confirmation.
# Uses weekly pivot points to establish long-term directional bias, then trades
# breakouts of the 6-hour Donchian channel in the direction of the bias.
# Volume confirmation ensures breakouts have institutional participation.
# Designed to work in both bull and bear markets by using weekly pivot bias
# as a trend filter that adapts to longer-term market structure.
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_pivot_points(high, low, close):
    """Calculate standard pivot points and support/resistance levels"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot bias (long-term trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot_points = np.full(len(weekly_close), np.nan)
    r3_points = np.full(len(weekly_close), np.nan)
    s3_points = np.full(len(weekly_close), np.nan)
    
    for i in range(len(weekly_close)):
        pivot, r1, r2, r3, s1, s2, s3 = calculate_pivot_points(
            weekly_high[i], weekly_low[i], weekly_close[i]
        )
        pivot_points[i] = pivot
        r3_points[i] = r3
        s3_points[i] = s3
    
    # Align weekly pivot data to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_points)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3_points)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3_points)
    
    # Determine weekly bias: price above pivot = bullish bias, below = bearish bias
    weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close)
    weekly_bias = np.where(weekly_close_aligned > pivot_aligned, 1,  # bullish
                          np.where(weekly_close_aligned < pivot_aligned, -1, 0))  # bearish
    
    # Donchian channel on 6h data (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume filter: volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback - 1, 20)  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Get current weekly bias
        bias = weekly_bias[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i]
        breakout_down = close[i] < lowest_low[i]
        
        # Entry conditions with volume confirmation and bias filter
        # Only take long breakouts in bullish bias, short breakouts in bearish bias
        long_entry = breakout_up and bias == 1 and volume_filter[i]
        short_entry = breakout_down and bias == -1 and volume_filter[i]
        
        # Exit conditions: opposite Donchian breakout or loss of bias
        long_exit = (position == 1 and (breakout_down or bias == -1))
        short_exit = (position == -1 and (breakout_up or bias == 1))
        
        # Handle entries and exits
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
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Donchian_WeeklyPivotBias_Volume"
timeframe = "6h"
leverage = 1.0