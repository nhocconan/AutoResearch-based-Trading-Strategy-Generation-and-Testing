#!/usr/bin/env python3
"""
1d_1w_Camarilla_Breakout_WeeklyTrend_v8
Hypothesis: Use weekly trend filter (price above/below weekly 50 EMA) with daily Camarilla breakout and volume confirmation (1.5x avg). Long when weekly trend up and price breaks daily R3; short when weekly trend down and price breaks daily S3. Exit at daily pivot. Target: 10-20 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_Breakout_WeeklyTrend_v8"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA(50) for trend
    if len(close_1w) >= 50:
        close_1w_series = pd.Series(close_1w)
        ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    else:
        ema_50_1w = np.full(len(close_1w), np.nan)
    
    # Align to daily timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume average (20-period) for confirmation
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
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Weekly trend: price above/below weekly EMA(50)
        weekly_up = close[i] > ema_50_1w_aligned[i]
        weekly_down = close[i] < ema_50_1w_aligned[i]
        
        # Breakout entries with volume and trend filters
        long_setup = weekly_up and vol_confirm and (close[i] > r3_1d_aligned[i])
        short_setup = weekly_down and vol_confirm and (close[i] < s3_1d_aligned[i])
        
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