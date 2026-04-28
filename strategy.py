#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate daily pivot points (P, S1, S2, S3, R1, R2, R3)
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    pivot_d = (high_d + low_d + close_d) / 3
    r1_d = 2 * pivot_d - low_d
    s1_d = 2 * pivot_d - high_d
    r2_d = pivot_d + (high_d - low_d)
    s2_d = pivot_d - (high_d - low_d)
    r3_d = high_d + 2 * (pivot_d - low_d)
    s3_d = low_d - 2 * (high_d - pivot_d)
    
    # Align to 12h timeframe
    r3_d_aligned = align_htf_to_ltf(prices, df_1d, r3_d)
    s3_d_aligned = align_htf_to_ltf(prices, df_1d, s3_d)
    r1_d_aligned = align_htf_to_ltf(prices, df_1d, r1_d)
    s1_d_aligned = align_htf_to_ltf(prices, df_1d, s1_d)
    
    # Volume filter: above average volume (30-period)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_d_aligned[i]) or np.isnan(s3_d_aligned[i]) or 
            np.isnan(r1_d_aligned[i]) or np.isnan(s1_d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: above average volume
        vol_filter = volume[i] > vol_ma[i]
        
        # Entry conditions: 
        # Long: break above daily S3 with volume
        # Short: break below daily R3 with volume
        long_breakout = close[i] > s3_d_aligned[i]
        short_breakout = close[i] < r3_d_aligned[i]
        
        long_entry = long_breakout and vol_filter
        short_entry = short_breakout and vol_filter
        
        # Exit conditions: opposite S1/R1 level touch
        long_exit = (close[i] < s1_d_aligned[i]) and position == 1
        short_exit = (close[i] > r1_d_aligned[i]) and position == -1
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_DailyPivot_S3_R3_Breakout_Volume_Session"
timeframe = "12h"
leverage = 1.0