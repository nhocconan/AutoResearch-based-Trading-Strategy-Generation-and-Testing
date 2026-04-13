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
    
    # Get 1d data for weekly pivot and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate weekly high/low/close for pivot points
    week_high = np.full(len(high_1d), np.nan)
    week_low = np.full(len(low_1d), np.nan)
    week_close = np.full(len(close_1d), np.nan)
    week_volume = np.full(len(volume_1d), np.nan)
    
    for i in range(7, len(high_1d)):
        week_high[i] = np.max(high_1d[i-7:i])
        week_low[i] = np.min(low_1d[i-7:i])
        week_close[i] = close_1d[i-1]  # Previous day's close as weekly close
        week_volume[i] = np.sum(volume_1d[i-7:i])  # Sum of volume over past week
    
    # Calculate weekly pivot points and support/resistance levels
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
    
    # Get 60-period average volume on 1d for volume confirmation
    avg_volume_60d = np.full(len(volume_1d), np.nan)
    for i in range(60, len(volume_1d)):
        avg_volume_60d[i] = np.mean(volume_1d[i-60:i])
    
    # Get 6h data for price action
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 10:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Align all indicators to original timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    avg_volume_60d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_60d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(avg_volume_60d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 60-day average volume
        vol_confirm = volume[i] > avg_volume_60d_aligned[i]
        
        # Price action: close above/below pivot
        above_pivot = close[i] > pivot_aligned[i]
        below_pivot = close[i] < pivot_aligned[i]
        
        # Entry conditions: price rejection at S3/R3 with volume
        long_entry = (close[i] <= s3_aligned[i] * 1.02) and vol_confirm and above_pivot
        short_entry = (close[i] >= r3_aligned[i] * 0.98) and vol_confirm and below_pivot
        
        # Exit conditions: price crosses pivot
        exit_long = position == 1 and below_pivot
        exit_short = position == -1 and above_pivot
        
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

name = "6h_1d_weekly_pivot_rejection_volume"
timeframe = "6h"
leverage = 1.0