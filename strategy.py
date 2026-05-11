#!/usr/bin/env python3
name = "6h_WeeklyPivot_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for weekly pivot and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Weekly pivot points from 1d data (use last 5 days)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot (using last 5 days of 1d data)
    def calculate_weekly_pivot(high_arr, low_arr, close_arr):
        if len(high_arr) < 5:
            return np.nan, np.nan, np.nan
        # Use last 5 days
        hh = np.max(high_arr[-5:])
        ll = np.min(low_arr[-5:])
        cc = close_arr[-1]
        pivot = (hh + ll + cc) / 3.0
        r1 = 2 * pivot - ll
        s1 = 2 * pivot - hh
        return pivot, r1, s1
    
    # Calculate weekly pivot for each point (using expanding window)
    weekly_pivot = np.full(len(close_1d), np.nan)
    weekly_r1 = np.full(len(close_1d), np.nan)
    weekly_s1 = np.full(len(close_1d), np.nan)
    
    for i in range(4, len(close_1d)):  # Start from 5th day
        pivot, r1, s1 = calculate_weekly_pivot(high_1d[:i+1], low_1d[:i+1], close_1d[:i+1])
        weekly_pivot[i] = pivot
        weekly_r1[i] = r1
        weekly_s1[i] = s1
    
    # Daily trend: EMA20 vs EMA50
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = ema20_1d > ema50_1d
    
    # Align to 6h
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend)
    
    # Volume confirmation on 6h: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA and calculations
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or np.isnan(daily_uptrend_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly pivot + daily uptrend + volume confirmation
            if close[i] > weekly_pivot_aligned[i] and daily_uptrend_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly pivot + daily downtrend + volume confirmation
            elif close[i] < weekly_pivot_aligned[i] and not daily_uptrend_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below weekly pivot OR daily trend turns down
            if close[i] < weekly_pivot_aligned[i] or not daily_uptrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above weekly pivot OR daily trend turns up
            if close[i] > weekly_pivot_aligned[i] or daily_uptrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals