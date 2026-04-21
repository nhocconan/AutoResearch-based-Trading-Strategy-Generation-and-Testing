#!/usr/bin/env python3
"""
6h_1w_1d_Pivot_R3S3_Fade_Reverse_v1
Hypothesis: Fade at weekly R3/S3 and daily R3/S3 levels in counter-trend direction. Long at daily S3 with weekly bias up, short at daily R3 with weekly bias down. Uses weekly trend filter (price > weekly SMA50 for long bias, < for short bias) to avoid counter-trend traps. Works in both bull/bear by aligning with weekly trend while exploiting daily mean reversion at extreme pivot levels. Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly SMA50 for trend filter
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Load daily data for Pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Pivot calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Pivot point and support/resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    r3 = prev_high + 2 * (pivot - prev_low)
    s3 = prev_low - 2 * (prev_high - pivot)
    r4 = prev_high + 3 * (pivot - prev_low)
    s4 = prev_low - 3 * (prev_high - pivot)
    
    # Align to 6h timeframe
    r3_daily_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_daily_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r3_weekly_aligned = align_htf_to_ltf(prices, df_1w, 
                                          np.roll(df_1w['high'].values, 1) + 
                                          2 * (pd.Series(df_1w['close'].values).rolling(50, min_periods=50).mean().values - 
                                               np.roll(df_1w['low'].values, 1)))
    s3_weekly_aligned = align_htf_to_ltf(prices, df_1w, 
                                         np.roll(df_1w['low'].values, 1) - 
                                         2 * (np.roll(df_1w['high'].values, 1) - 
                                              pd.Series(df_1w['close'].values).rolling(50, min_periods=50).mean().values))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(sma_50_1w_aligned[i]) or np.isnan(r3_daily_aligned[i]) or 
            np.isnan(s3_daily_aligned[i]) or np.isnan(r3_weekly_aligned[i]) or 
            np.isnan(s3_weekly_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        # Weekly trend filter: price > weekly SMA50 = bullish bias, < = bearish bias
        weekly_bullish = price > sma_50_1w_aligned[i]
        weekly_bearish = price < sma_50_1w_aligned[i]
        
        if position == 0:
            # Long at daily S3 with weekly bullish bias
            if (abs(price - s3_daily_aligned[i]) < 0.002 * s3_daily_aligned[i] and  # near S3
                weekly_bullish):
                signals[i] = 0.25
                position = 1
            # Short at daily R3 with weekly bearish bias
            elif (abs(price - r3_daily_aligned[i]) < 0.002 * r3_daily_aligned[i] and  # near R3
                  weekly_bearish):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: reverse at weekly S3 or reach daily R3
            if price <= s3_weekly_aligned[i] or price >= r3_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: reverse at weekly R3 or reach daily S3
            if price >= r3_weekly_aligned[i] or price <= s3_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1w_1d_Pivot_R3S3_Fade_Reverse_v1"
timeframe = "6h"
leverage = 1.0