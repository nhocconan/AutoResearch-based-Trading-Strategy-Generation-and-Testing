#!/usr/bin/env python3
"""
1d_WeeklyPivot_Breakout_1wTrend_Filter
Hypothesis: Focus on weekly pivot point breakouts with weekly trend filter on daily timeframe.
Targets 10-25 trades/year by requiring weekly pivot breakout in direction of weekly trend.
Works in bull markets (breakouts above pivot in uptrend) and bear markets (breakdowns below pivot in downtrend).
Weekly pivot provides institutional reference levels; weekly trend filter avoids counter-trend trades.
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
    
    # Get weekly data for pivot and trend
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week)
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    # Weekly pivot point (P) = (H + L + C) / 3
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    # Weekly resistance 1 (R1) = (2 * P) - L
    weekly_r1 = (2 * weekly_pivot) - prev_week_low
    # Weekly support 1 (S1) = (2 * P) - H
    weekly_s1 = (2 * weekly_pivot) - prev_week_high
    
    # Weekly trend: price > weekly pivot = uptrend, < weekly pivot = downtrend
    weekly_trend_up = prev_week_close > weekly_pivot
    weekly_trend_down = prev_week_close < weekly_pivot
    
    # Align weekly data to daily
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(weekly_trend_up_aligned[i]) or 
            np.isnan(weekly_trend_down_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: breakout in direction of weekly trend
        # Long: price breaks above weekly R1 + weekly uptrend
        long_entry = (close[i] > weekly_r1_aligned[i] and weekly_trend_up_aligned[i])
        # Short: price breaks below weekly S1 + weekly downtrend
        short_entry = (close[i] < weekly_s1_aligned[i] and weekly_trend_down_aligned[i])
        
        # Exit: price crosses weekly pivot in opposite direction
        long_exit = close[i] < weekly_pivot_aligned[i]
        short_exit = close[i] > weekly_pivot_aligned[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WeeklyPivot_Breakout_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0