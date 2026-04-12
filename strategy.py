#!/usr/bin/env python3
"""
4h_1d_1w_Camarilla_Breakout_Volume_Trend_v5
Hypothesis: Trade breakouts of daily R3/S3 only when weekly trend is aligned (price above/below weekly pivot) and volume > 1.3x average. Exit at opposite S3/R3 level. Target: 20-30 trades/year. Works in bull (follow weekly uptrend) and bear (follow weekly downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_Camarilla_Breakout_Volume_Trend_v5"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot calculation
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels (daily)
    r3_1d = close_1d + range_1d * 1.1
    s3_1d = close_1d - range_1d * 1.1
    
    # === WEEKLY DATA ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot calculation
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    
    # === VOLUME AVERAGE (20-period for 4h) ===
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    # Align to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(pivot_1w_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.3x average
        vol_confirm = volume[i] > 1.3 * vol_avg[i]
        
        # Weekly trend filter: price above/below weekly pivot
        weekly_uptrend = close[i] > pivot_1w_aligned[i]
        weekly_downtrend = close[i] < pivot_1w_aligned[i]
        
        # Breakout entries with volume and weekly trend alignment
        long_setup = (close[i] > r3_1d_aligned[i]) and vol_confirm and weekly_uptrend
        short_setup = (close[i] < s3_1d_aligned[i]) and vol_confirm and weekly_downtrend
        
        # Exit when price reaches opposite S3/R3 level (mean reversion to weekly context)
        exit_long = close[i] < s3_1d_aligned[i]
        exit_short = close[i] > r3_1d_aligned[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
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