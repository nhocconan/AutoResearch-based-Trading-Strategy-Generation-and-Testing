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
    
    # Get 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly pivot points from 1d data (using last week's data)
    # We need the previous week's high, low, close to calculate current week's pivot
    # Since we're using 1d data, we'll calculate weekly pivot based on the last 5 trading days
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly high, low, close using rolling window of 5 days (1 week)
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    # Calculate weekly pivot point and support/resistance levels
    # Pivot = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # R1 = 2*P - L
    weekly_r1 = 2 * weekly_pivot - weekly_low
    # S1 = 2*P - H
    weekly_s1 = 2 * weekly_pivot - weekly_high
    # R2 = P + (H - L)
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    # S2 = P - (H - L)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    # R3 = H + 2*(P - L)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    # S3 = L - 2*(H - P)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_6h = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_6h = align_htf_to_ltf(prices, df_1d, weekly_s1)
    weekly_r2_6h = align_htf_to_ltf(prices, df_1d, weekly_r2)
    weekly_s2_6h = align_htf_to_ltf(prices, df_1d, weekly_s2)
    weekly_r3_6h = align_htf_to_ltf(prices, df_1d, weekly_r3)
    weekly_s3_6h = align_htf_to_ltf(prices, df_1d, weekly_s3)
    
    # Get 1w trend filter (EMA34)
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_1w = close_1w > ema34_1w
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(weekly_pivot_6h[i]) or np.isnan(weekly_r3_6h[i]) or np.isnan(weekly_s3_6h[i]) or
            np.isnan(trend_up_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above weekly S3 + 1w uptrend + volume confirmation
            if close[i] > weekly_s3_6h[i] and trend_up_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly R3 + 1w downtrend + volume confirmation
            elif close[i] < weekly_r3_6h[i] and not trend_up_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price below weekly S1 OR 1w trend turns down
            if close[i] < weekly_s1_6h[i] or not trend_up_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price above weekly R1 OR 1w trend turns up
            if close[i] > weekly_r1_6h[i] or trend_up_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals