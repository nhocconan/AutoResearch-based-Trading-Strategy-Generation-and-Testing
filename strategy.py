#!/usr/bin/env python3
"""
6h_WeeklyPivot_DailyTrend_Breakout
Hypothesis: Use weekly pivot points to identify key weekly support/resistance levels.
Go long when price breaks above weekly R1 with daily EMA50 uptrend.
Go short when price breaks below weekly S1 with daily EMA50 downtrend.
Weekly pivots provide strong institutional support/resistance that works across market regimes.
Daily EMA50 filter ensures trades align with intermediate-term trend.
Designed for low trade frequency (~15-30/year) with high win rate by requiring confluence of weekly structure and daily trend.
Works in both bull and bear markets by following the daily trend direction and using weekly structure as filter.
"""

name = "6h_WeeklyPivot_DailyTrend_Breakout"
timeframe = "6h"
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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    # Using standard pivot point calculation: P = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    prev_weekly_close = df_1w['close'].shift(1).values
    
    # Calculate pivot point and support/resistance levels
    pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    r1 = 2 * pivot - prev_weekly_low
    s1 = 2 * pivot - prev_weekly_high
    
    # Align to 6h timeframe
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA and pivot calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or np.isnan(ema_50_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price breakout conditions
        breakout_above_r1 = close[i] > r1_6h[i]
        breakdown_below_s1 = close[i] < s1_6h[i]
        
        trend_up = close[i] > ema_50_6h[i]
        trend_down = close[i] < ema_50_6h[i]
        
        if position == 0:
            # Long: break above weekly R1 + daily uptrend
            if breakout_above_r1 and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1 + daily downtrend
            elif breakdown_below_s1 and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below weekly S1 or trend reversal
            if close[i] < s1_6h[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above weekly R1 or trend reversal
            if close[i] > r1_6h[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals