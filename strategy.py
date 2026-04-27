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
    
    # Get daily data for weekly pivot calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week's data
    # Use last complete week: Monday to Friday of previous week
    high_weekly = df_1d['high'].values
    low_weekly = df_1d['low'].values
    close_weekly = df_1d['close'].values
    
    # Weekly pivot calculation: need 5 days of data
    # We'll use rolling window of 5 days to get weekly high/low/close
    week_high = pd.Series(high_weekly).rolling(window=5, min_periods=5).max().values
    week_low = pd.Series(low_weekly).rolling(window=5, min_periods=5).min().values
    week_close = pd.Series(close_weekly).rolling(window=5, min_periods=5).last().values
    
    # Calculate pivot points: P = (H+L+C)/3
    weekly_pivot = (week_high + week_low + week_close) / 3.0
    # Weekly resistance/support levels
    weekly_r1 = 2 * weekly_pivot - week_low
    weekly_s1 = 2 * weekly_pivot - week_high
    weekly_r2 = weekly_pivot + (week_high - week_low)
    weekly_s2 = weekly_pivot - (week_high - week_low)
    weekly_r3 = week_high + 2 * (weekly_pivot - week_low)
    weekly_s3 = week_low - 2 * (week_high - weekly_pivot)
    weekly_r4 = weekly_pivot + 3 * (week_high - week_low)
    weekly_s4 = weekly_pivot - 3 * (week_high - week_low)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, weekly_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, weekly_s4)
    
    # Volume filter: volume > 1.8x 20-period average (stricter for 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above weekly R4 with volume
            if (close[i] > r4_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below weekly S4 with volume
            elif (close[i] < s4_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price falls back below weekly R3 (failed breakout)
            if close[i] < r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises back above weekly S3 (failed breakdown)
            if close[i] > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R4S4_Breakout_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0