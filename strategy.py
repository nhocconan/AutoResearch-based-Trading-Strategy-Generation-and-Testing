# 6H_WEEKLYPIVOT_CAMARILLA_TREND_REVERSAL
# Hypothesis: In 6h timeframe, combine weekly pivot points with daily Camarilla levels for trend reversal signals.
# Uses weekly pivot as trend filter and daily Camarilla R3/S3 for reversal entries, reducing trades while capturing
# major reversals in both bull and bear markets. Target: 50-150 total trades over 4 years.
# Weekly pivot provides higher timeframe context; daily Camarilla provides precise entry/exit levels.

name = "6H_WEEKLYPIVOT_CAMARILLA_TREND_REVERSAL"
timeframe = "6h"
leverage = 1.0

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
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (using prior day's OHLC)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    camarilla_range = daily_high - daily_low
    r3 = daily_close + (camarilla_range * 1.1 / 4)  # R3 level
    s3 = daily_close - (camarilla_ratio * 1.1 / 4)  # S3 level
    
    # Calculate weekly pivot points (using prior week's OHLC)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot_point = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot_point - weekly_low
    s1 = 2 * pivot_point - weekly_high
    
    # Align indicators to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3, additional_delay_bars=1)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3, additional_delay_bars=1)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_point)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Weekly trend filter (price relative to weekly pivot)
    weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(weekly_close_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        weekly_trend_up = weekly_close_aligned[i] > pivot_aligned[i]
        weekly_trend_down = weekly_close_aligned[i] < pivot_aligned[i]
        
        if position == 0:
            # Long reversal: price drops to S3 in weekly uptrend
            if (weekly_trend_up and close[i] <= s3_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short reversal: price rises to R3 in weekly downtrend
            elif (weekly_trend_down and close[i] >= r3_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reaches weekly R1 or reverses below pivot
            if close[i] >= r1_aligned[i] or weekly_close_aligned[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches weekly S1 or reverses above pivot
            if close[i] <= s1_aligned[i] or weekly_close_aligned[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals