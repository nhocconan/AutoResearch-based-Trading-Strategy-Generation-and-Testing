#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_WeeklyPivot_CongestionBreakout"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for congestion detection
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly range and congestion detection
    weekly_range = high_1w - low_1w
    range_ma = pd.Series(weekly_range).rolling(window=4, min_periods=4).mean().values
    congestion = weekly_range < (0.6 * range_ma)
    
    # Daily data for price levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close for pivot calculation
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Standard pivot point
    pivot = (prev_high + prev_low + prev_close) / 3.0
    # Support and resistance levels
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # Align to 12h timeframe
    congestion_aligned = align_htf_to_ltf(prices, df_1w, congestion)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(congestion_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R1 during weekly congestion
            long_cond = (close[i] > r1_aligned[i] and 
                        congestion_aligned[i] and
                        volume_filter[i])
            
            # Short: break below S1 during weekly congestion
            short_cond = (close[i] < s1_aligned[i] and 
                         congestion_aligned[i] and
                         volume_filter[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to pivot or congestion ends
            if (close[i] < pivot_aligned[i]) or (not congestion_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to pivot or congestion ends
            if (close[i] > pivot_aligned[i]) or (not congestion_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals