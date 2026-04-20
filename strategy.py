#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_Pivot_R3S3_Breakout_Volume_TrendFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for weekly and daily pivots
        return np.zeros(n)
    
    # Get weekly and daily data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 10 or len(df_1d) < 20:
        return np.zeros(n)
    
    # === Weekly: Calculate pivot points (using previous week's data) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Use previous week's OHLC for current week's pivot
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    
    # Set first week's values to NaN
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    # Calculate pivot points and R3/S3 levels
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    r1_1w = 2 * pivot_1w - prev_low_1w
    s1_1w = 2 * pivot_1w - prev_high_1w
    r2_1w = pivot_1w + (prev_high_1w - prev_low_1w)
    s2_1w = pivot_1w - (prev_high_1w - prev_low_1w)
    r3_1w = prev_high_1w + 2 * (pivot_1w - prev_low_1w)
    s3_1w = prev_low_1w - 2 * (prev_high_1w - pivot_1w)
    
    # === Daily: Calculate pivot points (using previous day's data) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use previous day's OHLC for current day's pivot
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # Set first day's values to NaN
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Calculate pivot points and R3/S3 levels
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    r1_1d = 2 * pivot_1d - prev_low_1d
    s1_1d = 2 * pivot_1d - prev_high_1d
    r2_1d = pivot_1d + (prev_high_1d - prev_low_1d)
    s2_1d = pivot_1d - (prev_high_1d - prev_low_1d)
    r3_1d = prev_high_1d + 2 * (pivot_1d - prev_low_1d)
    s3_1d = prev_low_1d - 2 * (prev_high_1d - pivot_1d)
    
    # Align weekly and daily pivot levels to 6h timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # === 6h: Volume ratio (current vs 20-period average) ===
    close = prices['close'].values
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Get values
        close_val = close[i]
        r3_1w_val = r3_1w_aligned[i]
        s3_1w_val = s3_1w_aligned[i]
        r3_1d_val = r3_1d_aligned[i]
        s3_1d_val = s3_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(r3_1w_val) or np.isnan(s3_1w_val) or 
            np.isnan(r3_1d_val) or np.isnan(s3_1d_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly R3 or daily R3 with volume confirmation
            if ((close_val > r3_1w_val or close_val > r3_1d_val) and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S3 or daily S3 with volume confirmation
            elif ((close_val < s3_1w_val or close_val < s3_1d_val) and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price drops below weekly S3 or daily S3
            if close_val < s3_1w_val or close_val < s3_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises above weekly R3 or daily R3
            if close_val > r3_1w_val or close_val > r3_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals