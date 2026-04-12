#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_Breakout_Volume_Regime
Hypothesis: On 1h timeframe, enter long when price breaks above 4h Camarilla R3 with volume confirmation (>1.5x average), short when breaks below 4h S3. Uses daily trend filter and session filter (08-20 UTC) to reduce noise. Target: 60-150 total trades over 4 years = 15-37/year for 1h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_Camarilla_Breakout_Volume_Regime"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4H CAMARILLA PIVOT LEVELS ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h pivot calculation
    pivot_4h = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    
    # Camarilla levels (4h)
    r3_4h = close_4h + range_4h * 1.1
    s3_4h = close_4h - range_4h * 1.1
    
    # === DAILY TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot calculation
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Daily Camarilla levels for trend filter
    r3_1d = close_1d + range_1d * 1.1
    s3_1d = close_1d - range_1d * 1.1
    
    # Align to 1h timeframe
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Volume average (48-period for 1h = ~2 days) for confirmation
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 48:
            vol_sum -= volume[i-48]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # start after warmup
        # Skip if indicators not available or outside session
        if (np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or 
            np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            vol_avg[i] == 0.0 or hours[i] < 8 or hours[i] > 20):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Trend filter: use daily context
        # Only go long if above daily S3, only short if below daily R3
        long_allowed = close[i] > s3_1d_aligned[i]
        short_allowed = close[i] < r3_1d_aligned[i]
        
        # Breakout entries at S3/R3 with volume and trend filter
        long_setup = (close[i] > r3_4h_aligned[i]) and vol_confirm and long_allowed
        short_setup = (close[i] < s3_4h_aligned[i]) and vol_confirm and short_allowed
        
        # Exit when price returns to 4h pivot (mean reversion)
        pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
        exit_long = close[i] < pivot_4h_aligned[i]
        exit_short = close[i] > pivot_4h_aligned[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.20
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals