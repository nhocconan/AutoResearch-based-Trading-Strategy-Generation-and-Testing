#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate classic pivot points: P = (H + L + C)/3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Calculate support/resistance levels
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    r3_1d = high_1d + 2 * (pivot_1d - low_1d)
    s3_1d = low_1d - 2 * (high_1d - pivot_1d)
    
    # Align pivot levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Volume filter: current volume > 1.5 * 24-period average (approx 6 days)
    volume_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # Need pivot data and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_6h[i]) or 
            np.isnan(r1_6h[i]) or 
            np.isnan(s1_6h[i]) or 
            np.isnan(r2_6h[i]) or 
            np.isnan(s2_6h[i]) or 
            np.isnan(r3_6h[i]) or 
            np.isnan(s3_6h[i]) or 
            np.isnan(volume_ma24[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma24[i])
        
        if position == 0:
            # Long: price breaks above R1 with volume
            if (close[i] > r1_6h[i] and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume
            elif (close[i] < s1_6h[i] and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below pivot
            if close[i] < pivot_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above pivot
            if close[i] > pivot_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_R1_S1_Breakout_Volume"
timeframe = "6h"
leverage = 1.0