#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points using previous day's data
    pivot = np.full(len(high_1d), np.nan)
    r1 = np.full(len(high_1d), np.nan)
    s1 = np.full(len(high_1d), np.nan)
    r2 = np.full(len(high_1d), np.nan)
    s2 = np.full(len(high_1d), np.nan)
    r3 = np.full(len(high_1d), np.nan)
    s3 = np.full(len(high_1d), np.nan)
    
    for i in range(1, len(high_1d)):
        pivot[i] = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
        r1[i] = 2 * pivot[i] - low_1d[i-1]
        s1[i] = 2 * pivot[i] - high_1d[i-1]
        r2[i] = pivot[i] + (high_1d[i-1] - low_1d[i-1])
        s2[i] = pivot[i] - (high_1d[i-1] - low_1d[i-1])
        r3[i] = high_1d[i-1] + 2 * (pivot[i] - low_1d[i-1])
        s3[i] = low_1d[i-1] - 2 * (high_1d[i-1] - pivot[i])
    
    # Calculate 6-period EMA on 6h close for trend filter
    close_series = pd.Series(close)
    ema6 = close_series.ewm(span=6, adjust=False, min_periods=6).mean().values
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(ema6[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA6
        uptrend = close[i] > ema6[i]
        downtrend = close[i] < ema6[i]
        
        # Entry conditions: price touches S1/S2/S3 in uptrend or R1/R2/R3 in downtrend
        long_entry = uptrend and (close[i] <= s1_aligned[i] * 1.002 or close[i] <= s2_aligned[i] * 1.002 or close[i] <= s3_aligned[i] * 1.002)
        short_entry = downtrend and (close[i] >= r1_aligned[i] * 0.998 or close[i] >= r2_aligned[i] * 0.998 or close[i] >= r3_aligned[i] * 0.998)
        
        # Exit conditions: price crosses pivot or reaches opposite level
        exit_long = position == 1 and (close[i] >= pivot_aligned[i] or close[i] >= r1_aligned[i])
        exit_short = position == -1 and (close[i] <= pivot_aligned[i] or close[i] <= s1_aligned[i])
        
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

name = "6h_1d_pivot_reversion_ema_filter"
timeframe = "6h"
leverage = 1.0