#!/usr/bin/env python3
"""
1d_1w_Camarilla_Breakout_WeeklyTrend_v1
Hypothesis: On daily timeframe, enter long when price breaks above daily Camarilla R3 with weekly trend confirmation (price above weekly SMA50), enter short when price breaks below daily Camarilla S3 with weekly trend confirmation (price below weekly SMA50). Uses daily price channels for structure and weekly trend filter to avoid counter-trend trades. Designed for low trade frequency (<25/year) and robust performance in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_Breakout_WeeklyTrend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === DAILY CAMARILLA PIVOT LEVELS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla pivot levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla R3 and S3 levels (key reversal levels)
    r3 = close_1d + range_1d * 1.1 / 4
    s3 = close_1d - range_1d * 1.1 / 4
    
    # Align to daily timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === WEEKLY TREND FILTER (SMA50) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly SMA50
    sma50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        for i in range(50, len(close_1w)):
            sma50_1w[i] = np.mean(close_1w[i-50:i])
    
    # Align weekly close and SMA50 to daily timeframe
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    # Weekly trend: price above SMA50 = uptrend, below = downtrend
    weekly_uptrend = close_1w_aligned > sma50_1w_aligned
    weekly_downtrend = close_1w_aligned < sma50_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup for SMA50
        # Skip if indicators not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(sma50_1w_aligned[i]) or np.isnan(close_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with weekly trend filter
        long_breakout = (close[i] > r3_aligned[i]) and weekly_uptrend[i]
        short_breakout = (close[i] < s3_aligned[i]) and weekly_downtrend[i]
        
        # Exit conditions: reversal back inside opposite Camarilla level
        # Exit long if price breaks below S3
        exit_long = close[i] < s3_aligned[i]
        # Exit short if price breaks above R3
        exit_short = close[i] > r3_aligned[i]
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals