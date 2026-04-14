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
    
    # Load daily data once for pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Create arrays for alignment
    pivot_arr = pivot
    r1_arr = r1
    s1_arr = s1
    r2_arr = r2
    s2_arr = s2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20
    
    # Pre-compute session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(2, n):
        if not in_session[i]:
            continue
        
        # Get aligned daily data
        pivot_d = align_htf_to_ltf(prices, df_1d, pivot_arr)[i]
        r1_d = align_htf_to_ltf(prices, df_1d, r1_arr)[i]
        s1_d = align_htf_to_ltf(prices, df_1d, s1_arr)[i]
        r2_d = align_htf_to_ltf(prices, df_1d, r2_arr)[i]
        s2_d = align_htf_to_ltf(prices, df_1d, s2_arr)[i]
        
        if np.isnan(pivot_d) or np.isnan(r1_d) or np.isnan(s1_d) or np.isnan(r2_d) or np.isnan(s2_d):
            continue
        
        if position == 0:
            # Long: Price rejects S1 (daily support) with close above S1
            if close[i] > s1_d and low[i] <= s1_d * 1.002:
                position = 1
                signals[i] = position_size
            # Short: Price rejects R1 (daily resistance) with close below R1
            elif close[i] < r1_d and high[i] >= r1_d * 0.998:
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price reaches R2 (daily resistance target) or reverses at R1
            if close[i] >= r2_d or (close[i] < r1_d and high[i] >= r1_d * 0.998):
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price reaches S2 (daily support target) or reverses at S1
            if close[i] <= s2_d or (close[i] > s1_d and low[i] <= s1_d * 1.002):
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1h_DailyPivot_Rejection_v1"
timeframe = "1h"
leverage = 1.0