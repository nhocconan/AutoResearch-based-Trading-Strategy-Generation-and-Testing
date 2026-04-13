#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot calculation (weekly pivot using prior week)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using prior week's high, low, close)
    week_high = np.full(len(high_1d), np.nan)
    week_low = np.full(len(high_1d), np.nan)
    week_close = np.full(len(close_1d), np.nan)
    
    for i in range(7, len(high_1d)):
        week_high[i] = np.max(high_1d[i-7:i])
        week_low[i] = np.min(low_1d[i-7:i])
        week_close[i] = close_1d[i-1]  # Previous day's close as weekly close
    
    pivot = np.full(len(high_1d), np.nan)
    r1 = np.full(len(high_1d), np.nan)
    s1 = np.full(len(high_1d), np.nan)
    r2 = np.full(len(high_1d), np.nan)
    s2 = np.full(len(high_1d), np.nan)
    r3 = np.full(len(high_1d), np.nan)
    s3 = np.full(len(high_1d), np.nan)
    
    for i in range(7, len(high_1d)):
        if not (np.isnan(week_high[i]) or np.isnan(week_low[i]) or np.isnan(week_close[i])):
            pivot[i] = (week_high[i] + week_low[i] + week_close[i]) / 3.0
            r1[i] = 2 * pivot[i] - week_low[i]
            s1[i] = 2 * pivot[i] - week_high[i]
            r2[i] = pivot[i] + (week_high[i] - week_low[i])
            s2[i] = pivot[i] - (week_high[i] - week_low[i])
            r3[i] = week_high[i] + 2 * (pivot[i] - week_low[i])
            s3[i] = week_low[i] - 2 * (week_high[i] - pivot[i])
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    
    # Calculate 20-period average volume on 12h
    avg_volume_12h = np.full(len(volume_12h), np.nan)
    for i in range(20, len(volume_12h)):
        avg_volume_12h[i] = np.mean(volume_12h[i-20:i])
    
    # Align indicators to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    avg_volume_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_volume_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(avg_volume_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > average 12h volume
        vol_confirm = volume[i] > avg_volume_12h_aligned[i]
        
        # Pivot-based entry conditions
        # Long when price crosses above S3 with volume
        # Short when price crosses below R3 with volume
        long_entry = (close[i] > s3_aligned[i]) and vol_confirm and (close[i-1] <= s3_aligned[i-1])
        short_entry = (close[i] < r3_aligned[i]) and vol_confirm and (close[i-1] >= r3_aligned[i-1])
        
        # Exit conditions: price returns to pivot level
        exit_long = position == 1 and close[i] <= pivot_aligned[i]
        exit_short = position == -1 and close[i] >= pivot_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_1d_pivot_volume_breakout"
timeframe = "6h"
leverage = 1.0