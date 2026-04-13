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
    
    # Get 1d data for daily pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (using prior day data)
    pivot = np.full(len(high_1d), np.nan)
    r1 = np.full(len(high_1d), np.nan)
    s1 = np.full(len(high_1d), np.nan)
    r2 = np.full(len(high_1d), np.nan)
    s2 = np.full(len(high_1d), np.nan)
    
    for i in range(1, len(high_1d)):
        pivot[i] = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
        r1[i] = 2 * pivot[i] - low_1d[i-1]
        s1[i] = 2 * pivot[i] - high_1d[i-1]
        r2[i] = pivot[i] + (high_1d[i-1] - low_1d[i-1])
        s2[i] = pivot[i] - (high_1d[i-1] - low_1d[i-1])
    
    # Align pivot levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume ratio (current volume vs 20-period average)
    vol_avg = np.full(n, np.nan)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_avg[i] = vol_sum / vol_count
            vol_sum -= volume[i - 20]
            vol_count -= 1
        elif i >= 0:
            vol_avg[i] = vol_sum / vol_count
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Price near pivot levels (within 0.5% tolerance)
        near_pivot = (abs(close[i] - pivot_aligned[i]) / pivot_aligned[i] < 0.005)
        near_r1 = (abs(close[i] - r1_aligned[i]) / r1_aligned[i] < 0.005)
        near_s1 = (abs(close[i] - s1_aligned[i]) / s1_aligned[i] < 0.005)
        near_r2 = (abs(close[i] - r2_aligned[i]) / r2_aligned[i] < 0.005)
        near_s2 = (abs(close[i] - s2_aligned[i]) / s2_aligned[i] < 0.005)
        
        # Entry conditions: price near support/resistance with volume
        long_entry = vol_confirm and (near_s1 or near_s2)
        short_entry = vol_confirm and (near_r1 or near_r2)
        
        # Exit conditions: price moves away from pivot or opposite level touched
        exit_long = position == 1 and (near_r1 or near_r2 or abs(close[i] - pivot_aligned[i]) / pivot_aligned[i] > 0.02)
        exit_short = position == -1 and (near_s1 or near_s2 or abs(close[i] - pivot_aligned[i]) / pivot_aligned[i] > 0.02)
        
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

name = "4h_1d_pivot_reversion_volume"
timeframe = "4h"
leverage = 1.0