#!/usr/bin/env python3
"""
6h_1w_1d_Weekly_Pivot_Momentum_v1
Hypothesis: Combine weekly pivot levels (1w) with daily momentum (1d) on 6h timeframe.
Long when price breaks above weekly R4 with daily momentum confirmation.
Short when price breaks below weekly S4 with daily momentum confirmation.
Uses weekly structure for direction and daily momentum for entry timing.
Targets 50-150 total trades over 4 years to minimize fee drag.
Works in bull via breakout continuation, in bear via breakdown continuation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_Weekly_Pivot_Momentum_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's OHLC for weekly pivot calculation
    prev_week_high = df_1w['high'].iloc[-2] if len(df_1w) >= 2 else df_1w['high'].iloc[-1]
    prev_week_low = df_1w['low'].iloc[-2] if len(df_1w) >= 2 else df_1w['low'].iloc[-1]
    prev_week_close = df_1w['close'].iloc[-2] if len(df_1w) >= 2 else df_1w['close'].iloc[-1]
    
    # Calculate weekly pivot levels
    week_range = prev_week_high - prev_week_low
    if week_range <= 0:
        return np.zeros(n)
    
    # Weekly pivot levels: R4, R3, S3, S4
    weekly_r4 = prev_week_close + week_range * 1.1
    weekly_r3 = prev_week_close + week_range * 1.1 / 2
    weekly_s3 = prev_week_close - week_range * 1.1 / 2
    weekly_s4 = prev_week_close - week_range * 1.1
    
    # Align weekly levels to 6h timeframe
    weekly_r4_array = np.full(len(df_1w), weekly_r4)
    weekly_r3_array = np.full(len(df_1w), weekly_r3)
    weekly_s3_array = np.full(len(df_1w), weekly_s3)
    weekly_s4_array = np.full(len(df_1w), weekly_s4)
    
    weekly_r4_aligned = align_htf_to_ltf(prices, df_1w, weekly_r4_array)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3_array)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1w, weekly_s3_array)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_1w, weekly_s4_array)
    
    # Daily data for momentum confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Daily momentum: ROC(5) - rate of change over 5 days
    daily_close = df_1d['close'].values
    roc_5 = np.full_like(daily_close, np.nan)
    for i in range(5, len(daily_close)):
        roc_5[i] = (daily_close[i] - daily_close[i-5]) / daily_close[i-5] * 100
    
    # Align daily ROC to 6h timeframe
    roc_5_aligned = align_htf_to_ltf(prices, df_1d, roc_5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(weekly_r4_aligned[i]) or np.isnan(weekly_r3_aligned[i]) or
            np.isnan(weekly_s3_aligned[i]) or np.isnan(weekly_s4_aligned[i]) or
            np.isnan(roc_5_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with momentum confirmation
        long_breakout = high[i] > weekly_r4_aligned[i] and roc_5_aligned[i] > 0
        short_breakout = low[i] < weekly_s4_aligned[i] and roc_5_aligned[i] < 0
        
        # Exit conditions: return to weekly midpoint (mean reversion)
        weekly_midpoint = (weekly_r3 + weekly_s3) / 2
        weekly_midpoint_array = np.full(len(df_1w), weekly_midpoint)
        weekly_midpoint_aligned = align_htf_to_ltf(prices, df_1w, weekly_midpoint_array)
        
        long_exit = close[i] < weekly_midpoint_aligned[i]
        short_exit = close[i] > weekly_midpoint_aligned[i]
        
        # Signal logic
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals