#!/usr/bin/env python3
"""
6h Weekly Pivot Breakout with Volume Confirmation and Trend Filter
Long: Price breaks above weekly R3 + volume > 1.5x 6h volume SMA(20) + price > 6h EMA(50)
Short: Price breaks below weekly S3 + volume > 1.5x 6h volume SMA(20) + price < 6h EMA(50)
Exit: Opposite breakout or EMA cross
Uses weekly pivots for structural levels, volume for confirmation, EMA for trend filter.
Designed to capture strong breakouts in both bull and bear markets with institutional levels.
Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf  # Note: using align_ltf_to_htf for proper alignment

def calculate_pivot_points(high, low, close):
    """Calculate standard pivot points"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) == 0:
        return np.zeros(n)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points
    _, _, _, weekly_r3, _, _, weekly_s3 = calculate_pivot_points(
        weekly_high, weekly_low, weekly_close
    )
    
    # Align weekly R3 and S3 to 6h timeframe
    weekly_r3_aligned = align_ltf_to_htf(prices, df_weekly, weekly_r3)
    weekly_s3_aligned = align_ltf_to_htf(prices, df_weekly, weekly_s3)
    
    # Calculate 6h EMA(50) for trend filter
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6h volume SMA(20)
    volume_series = pd.Series(volume)
    vol_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(50, 20)  # need EMA and volume SMA
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_r3_aligned[i]) or np.isnan(weekly_s3_aligned[i]) or
            np.isnan(ema_50[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_20[i]
        ema_val = ema_50[i]
        r3_level = weekly_r3_aligned[i]
        s3_level = weekly_s3_aligned[i]
        
        if position == 0:
            # Long: Break above weekly R3 + volume spike + above EMA50
            if price > r3_level and vol > 1.5 * vol_sma_val and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly S3 + volume spike + below EMA50
            elif price < s3_level and vol > 1.5 * vol_sma_val and price < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Break below weekly S3 or EMA cross down
            if price < s3_level or price < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Break above weekly R3 or EMA cross up
            if price > r3_level or price > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R3S3_Breakout_Volume_EMA50"
timeframe = "6h"
leverage = 1.0