#!/usr/bin/env python3
"""
6h_1D_Camarilla_R3S3_Fade_With_Trend_Filter
Hypothesis: Trade 6-hour timeframe with fade strategy at Camarilla R3/S3 levels from 1-day timeframe, 
filtered by 1-day EMA20 trend direction. In bear markets, price often reverts from extreme daily 
levels (R3/S3) but continues in trend direction. Uses only 2 conditions: Camarilla level touch + trend filter.
Targets 50-150 total trades over 4 years (12-37/year) with position size 0.25.
Works in bull/bear: trend filter ensures we only fade against the trend when appropriate.
"""

name = "6h_1D_Camarilla_R3S3_Fade_With_Trend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 5:
        return np.zeros(n)
    
    # Calculate Camarilla levels for previous day (using OHLC from prior day)
    # Camarilla: R3 = close + (high - low) * 1.1/2, S3 = close - (high - low) * 1.1/2
    # We use previous day's OHLC to avoid look-ahead
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Calculate Camarilla R3 and S3 for each day
    camarilla_r3 = np.full_like(daily_close, np.nan)
    camarilla_s3 = np.full_like(daily_close, np.nan)
    
    for i in range(1, len(daily_close)):  # Start from 1 to use previous day
        high_low = daily_high[i-1] - daily_low[i-1]
        camarilla_r3[i] = daily_close[i-1] + high_low * 1.1 / 2
        camarilla_s3[i] = daily_close[i-1] - high_low * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (already uses previous day's values)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s3)
    
    # Calculate daily EMA20 for trend filter
    ema20_daily = np.full_like(daily_close, np.nan)
    if len(daily_close) >= 20:
        multiplier = 2.0 / (20 + 1)
        ema20_daily[19] = np.mean(daily_close[:20])
        for i in range(20, len(daily_close)):
            ema20_daily[i] = multiplier * daily_close[i] + (1 - multiplier) * ema20_daily[i-1]
    ema20_daily_aligned = align_htf_to_ltf(prices, df_daily, ema20_daily)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema20_daily_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long fade at S3: price touches S3 and daily trend is up (EMA20 rising)
            if close[i] <= camarilla_s3_aligned[i] and ema20_daily_aligned[i] > ema20_daily_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short fade at R3: price touches R3 and daily trend is down (EMA20 falling)
            elif close[i] >= camarilla_r3_aligned[i] and ema20_daily_aligned[i] < ema20_daily_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches midpoint or trend changes
            midpoint = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2
            if close[i] >= midpoint or ema20_daily_aligned[i] < ema20_daily_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches midpoint or trend changes
            midpoint = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2
            if close[i] <= midpoint or ema20_daily_aligned[i] > ema20_daily_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals