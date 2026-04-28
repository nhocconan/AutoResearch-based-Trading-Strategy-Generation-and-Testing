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
    
    # Get 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Weekly pivot points from prior week
    # Calculate prior week's high, low, close
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(5)  # Prior week's high
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(5)    # Prior week's low
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(5)  # Prior week's close
    
    # Calculate pivot and support/resistance levels
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # Align to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        # Long: price crosses above S3 with volume surge (mean reversion from extreme)
        long_entry = (close[i] > s3_aligned[i]) and (close[i-1] <= s3_aligned[i-1]) and volume_surge[i]
        # Short: price crosses below R3 with volume surge (mean reversion from extreme)
        short_entry = (close[i] < r3_aligned[i]) and (close[i-1] >= r3_aligned[i-1]) and volume_surge[i]
        
        # Exit conditions: return to pivot or opposite extreme
        long_exit = (close[i] < pivot_aligned[i]) or (close[i] > r3_aligned[i])
        short_exit = (close[i] > pivot_aligned[i]) or (close[i] < s3_aligned[i])
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_S3_R3_MeanReversion_Volume"
timeframe = "6h"
leverage = 1.0