#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_donchian_weekly_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h average volume (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 20 to ensure sufficient data for calculations
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_avg_20[i]) or np.isnan(pivot_1w_aligned[i]) or
            np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current 6h volume
        vol_current = volume[i]
        vol_surge = vol_current > 1.5 * vol_avg_20[i]  # 50% above average
        
        price = close[i]
        
        # Long when price breaks above weekly R3 with volume surge
        long_breakout = price > r3_1w_aligned[i]
        long_signal = long_breakout and vol_surge
        
        # Short when price breaks below weekly S3 with volume surge
        short_breakout = price < s3_1w_aligned[i]
        short_signal = short_breakout and vol_surge
        
        # Exit when price returns to weekly pivot (mean reversion to weekly mean)
        exit_long = price < pivot_1w_aligned[i]
        exit_short = price > pivot_1w_aligned[i]
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals