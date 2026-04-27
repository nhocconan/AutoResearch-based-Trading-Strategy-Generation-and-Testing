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
    
    # Get daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Weekly pivot points from previous week's OHLC
    # Need at least 5 days for previous week
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Previous week's OHLC (excluding current incomplete week)
    # Use data from 5-10 days ago to get complete previous week
    week_close = df_1d['close'].values[-6:-1]  # 5 days: -6 to -2 (exclusive -1)
    week_high = df_1d['high'].values[-6:-1]
    week_low = df_1d['low'].values[-6:-1]
    week_open = df_1d['open'].values[-6:-1]
    
    if len(week_close) < 5:
        return np.zeros(n)
    
    # Previous week's values
    prev_week_high = np.max(week_high)
    prev_week_low = np.min(week_low)
    prev_week_close = week_close[-1]  # Friday close
    prev_week_open = week_open[0]     # Monday open
    
    # Calculate weekly pivot points
    pp = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    r1 = 2 * pp - prev_week_low
    s1 = 2 * pp - prev_week_high
    r2 = pp + (prev_week_high - prev_week_low)
    s2 = pp - (prev_week_high - prev_week_low)
    r3 = prev_week_high + 2 * (pp - prev_week_low)
    s3 = prev_week_low - 2 * (prev_week_high - pp)
    
    # Align weekly pivots to 6h timeframe (they change only when new week starts)
    # Create arrays of pivot levels for each day
    pp_series = np.full(len(df_1d), np.nan)
    r1_series = np.full(len(df_1d), np.nan)
    s1_series = np.full(len(df_1d), np.nan)
    r2_series = np.full(len(df_1d), np.nan)
    s2_series = np.full(len(df_1d), np.nan)
    r3_series = np.full(len(df_1d), np.nan)
    s3_series = np.full(len(df_1d), np.nan)
    
    # For each day, calculate pivot based on previous week
    for i in range(4, len(df_1d)):  # Start from index 4 (5th day) to have 5-day lookback
        week_high = np.max(df_1d['high'].values[i-4:i+1])  # Previous 5 days including current
        week_low = np.min(df_1d['low'].values[i-4:i+1])
        week_close = df_1d['close'].values[i]
        
        pp_val = (week_high + week_low + week_close) / 3.0
        r1_val = 2 * pp_val - week_low
        s1_val = 2 * pp_val - week_high
        r2_val = pp_val + (week_high - week_low)
        s2_val = pp_val - (week_high - week_low)
        r3_val = week_high + 2 * (pp_val - week_low)
        s3_val = week_low - 2 * (week_high - pp_val)
        
        pp_series[i] = pp_val
        r1_series[i] = r1_val
        s1_series[i] = s1_val
        r2_series[i] = r2_val
        s2_series[i] = s2_val
        r3_series[i] = r3_val
        s3_series[i] = s3_val
    
    # Align pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_series)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_series)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_series)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_series)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_series)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_series)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_series)
    
    # Volume filter: require volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    # Session filter: 00-23 UTC (all hours for 6h timeframe)
    hour = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = np.ones(n, dtype=bool)  # Always active for 6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 20  # need 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above S1 with volume
            if (close[i] > s1_aligned[i] and close[i-1] <= s1_aligned[i-1] and
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below R1 with volume
            elif (close[i] < r1_aligned[i] and close[i-1] >= r1_aligned[i-1] and
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below S2 or reaches R2
            if (close[i] < s2_aligned[i] and close[i-1] >= s2_aligned[i-1]) or \
               (close[i] > r2_aligned[i] and close[i-1] <= r2_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above R2 or reaches S2
            if (close[i] > r2_aligned[i] and close[i-1] <= r2_aligned[i-1]) or \
               (close[i] < s2_aligned[i] and close[i-1] >= s2_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_S1R1_Breakout_S2R2_Exit_Volume"
timeframe = "6h"
leverage = 1.0