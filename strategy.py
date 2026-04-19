#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Pivot_R1S1_Breakout_Volume_Target_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Pivot and volume
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Pivot points (Standard) on 1d OHLC
    pivot = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    r1 = 2 * pivot - df_1d['low']
    s1 = 2 * pivot - df_1d['high']
    
    # Calculate average volume on 1d
    avg_volume_1d = df_1d['volume'].rolling(window=20, min_periods=20).mean()
    
    # Align to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(avg_volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: current 12h volume > 1.5x average 1d volume
        volume_filter = volume[i] > 1.5 * avg_volume_1d_aligned[i]
        
        if position == 0:
            # Long when price breaks above R1 with volume confirmation
            if (close[i] > r1_aligned[i] and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S1 with volume confirmation
            elif (close[i] < s1_aligned[i] and volume_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price returns to pivot or below
            if close[i] <= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price returns to pivot or above
            if close[i] >= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals