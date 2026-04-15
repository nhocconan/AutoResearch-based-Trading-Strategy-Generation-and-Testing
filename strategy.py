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
    
    # Weekly high/low for Donchian channel (20 weeks)
    df_1w = get_htf_data(prices, '1w')
    high_20w = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max()
    low_20w = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min()
    high_20w_aligned = align_htf_to_ltf(prices, df_1w, high_20w)
    low_20w_aligned = align_htf_to_ltf(prices, df_1w, low_20w)
    
    # Daily pivot points for trend direction
    df_1d = get_htf_data(prices, '1d')
    pivot = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    r1 = 2 * pivot - df_1d['low'].values
    s1 = 2 * pivot - df_1d['high'].values
    r2 = pivot + (df_1d['high'].values - df_1d['low'].values)
    s2 = pivot - (df_1d['high'].values - df_1d['low'].values)
    r3 = r1 + (df_1d['high'].values - df_1d['low'].values)
    s3 = s1 - (df_1d['high'].values - df_1d['low'].values)
    
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20w_aligned[i]) or np.isnan(low_20w_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: price breaks above weekly Donchian high AND above daily R3 (strong bullish)
        if close[i] > high_20w_aligned[i] and close[i] > r3_aligned[i] and volume[i] > vol_threshold[i]:
            signals[i] = 0.25
        
        # Short: price breaks below weekly Donchian low AND below daily S3 (strong bearish)
        elif close[i] < low_20w_aligned[i] and close[i] < s3_aligned[i] and volume[i] > vol_threshold[i]:
            signals[i] = -0.25
        
        # Exit: price returns to daily pivot area (mean reversion to fair value)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < pivot_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > pivot_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_WeeklyDonchian_DailyPivot_Volume"
timeframe = "6h"
leverage = 1.0