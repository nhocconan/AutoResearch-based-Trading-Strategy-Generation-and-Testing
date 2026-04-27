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
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (R1, S1, R2, S2, R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels for each day
    close_prev = close_1d[-1]  # Most recent close
    high_prev = high_1d[-1]    # Most recent high
    low_prev = low_1d[-1]      # Most recent low
    
    # Calculate Camarilla levels based on previous day
    range_val = high_prev - low_prev
    if range_val <= 0:
        r1 = s1 = r2 = s2 = r3 = s3 = close_prev
    else:
        r1 = close_prev + (range_val * 1.1 / 12)
        s1 = close_prev - (range_val * 1.1 / 12)
        r2 = close_prev + (range_val * 1.1 / 6)
        s2 = close_prev - (range_val * 1.1 / 6)
        r3 = close_prev + (range_val * 1.1 / 4)
        s3 = close_prev - (range_val * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    r1_array = np.full(len(df_1d), r1)
    s1_array = np.full(len(df_1d), s1)
    r2_array = np.full(len(df_1d), r2)
    s2_array = np.full(len(df_1d), s2)
    r3_array = np.full(len(df_1d), r3)
    s3_array = np.full(len(df_1d), s3)
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_array)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_array)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_array)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_array)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_array)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_array)
    
    # Volume filter: require volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 00-23 UTC (all hours)
    hour = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hour >= 0) & (hour <= 23)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 20  # need 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume
            if (close[i] > r1_aligned[i] and 
                volume_filter[i] and 
                session_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume
            elif (close[i] < s1_aligned[i] and 
                  volume_filter[i] and 
                  session_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below S1 (reversal)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 (reversal)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume"
timeframe = "4h"
leverage = 1.0