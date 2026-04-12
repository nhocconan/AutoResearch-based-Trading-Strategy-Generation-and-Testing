#!/usr/bin/env python3
"""
4h_1d_1w_Camarilla_Breakout_Enhanced_v8
Hypothesis: Increase win rate by requiring price to be above/below 4h EMA(20) for breakout direction alignment and tightening volume filter to 3.0x average. This reduces false breakouts while maintaining sufficient trades for statistical significance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_Camarilla_Breakout_Enhanced_v8"
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
    range_1w = high_1w - low_1w
    
    # Weekly Camarilla levels
    r3_1w = close_1w + range_1w * 1.1
    s3_1w = close_1w - range_1w * 1.1
    
    # === 4H EMA(20) FOR DIRECTION FILTER ===
    if len(close) >= 20:
        ema_20 = np.zeros_like(close)
        ema_20[0] = close[0]
        for i in range(1, len(close)):
            ema_20[i] = (close[i] * 0.0952) + (ema_20[i-1] * 0.9048)  # alpha = 2/(20+1)
    else:
        ema_20 = np.full_like(close, np.nan)
    
    # Align to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    ema_20_aligned = ema_20  # already in 4h timeframe
    
    # Volume average (20-period for 4h = ~10 hours) for confirmation
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 3.0x average (tight)
        vol_confirm = volume[i] > 3.0 * vol_avg[i]
        
        # Direction filter: price above/below EMA(20) for breakout alignment
        price_above_ema = close[i] > ema_20_aligned[i]
        price_below_ema = close[i] < ema_20_aligned[i]
        
        # Weekly range-bound context: price between S3 and R3
        weekly_range = (close[i] > s3_1w_aligned[i]) & (close[i] < r3_1w_aligned[i])
        
        # Breakout entries at S3/R3 with volume, direction, and weekly range filters
        long_setup = (close[i] > r3_1d_aligned[i]) and vol_confirm and price_above_ema and weekly_range
        short_setup = (close[i] < s3_1d_aligned[i]) and vol_confirm and price_below_ema and weekly_range
        
        # Exit when price returns to daily pivot (mean reversion)
        pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
        exit_long = close[i] < pivot_1d_aligned[i]
        exit_short = close[i] > pivot_1d_aligned[i]
        
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