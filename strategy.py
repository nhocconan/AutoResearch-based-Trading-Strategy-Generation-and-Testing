#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 300:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points
    week_high = np.full(len(high_1d), np.nan)
    week_low = np.full(len(high_1d), np.nan)
    week_close = np.full(len(close_1d), np.nan)
    
    for i in range(7, len(high_1d)):
        week_high[i] = np.max(high_1d[i-7:i])
        week_low[i] = np.min(low_1d[i-7:i])
        week_close[i] = close_1d[i-1]
    
    # Calculate pivot points and S3/R3 levels
    pivot = np.full(len(high_1d), np.nan)
    r3 = np.full(len(high_1d), np.nan)
    s3 = np.full(len(high_1d), np.nan)
    
    for i in range(7, len(high_1d)):
        if not (np.isnan(week_high[i]) or np.isnan(week_low[i]) or np.isnan(week_close[i])):
            pivot[i] = (week_high[i] + week_low[i] + week_close[i]) / 3.0
            r3[i] = week_high[i] + 2 * (pivot[i] - week_low[i])
            s3[i] = week_low[i] - 2 * (week_high[i] - pivot[i])
    
    # Align to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(200, n):
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Check volume condition
        vol_ma = np.mean(volume[max(0, i-10):i+1]) if i >= 10 else np.mean(volume[:i+1])
        vol_confirm = volume[i] > vol_ma
        
        # Entry conditions
        long_entry = close[i] > r3_aligned[i] and vol_confirm
        short_entry = close[i] < s3_aligned[i] and vol_confirm
        
        # Exit conditions
        exit_long = position == 1 and close[i] < pivot_aligned[i]
        exit_short = position == -1 and close[i] > pivot_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_pivot_breakout_volume"
timeframe = "12h"
leverage = 1.0