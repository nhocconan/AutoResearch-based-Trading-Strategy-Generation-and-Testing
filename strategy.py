#!/usr/bin/env python3
# 6h_WeeklyPivot_Breakout_12hTrend_Volume
# Hypothesis: Fade at weekly R3/S3 and breakout at R4/S4 with 12h trend filter and volume confirmation.
# Weekly pivot levels act as strong support/resistance in both bull/bear markets. The 12h trend ensures
# we trade with higher timeframe momentum, while volume confirmation reduces false breakouts. Target: 15-30 trades/year.

name = "6h_WeeklyPivot_Breakout_12hTrend_Volume"
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
    volume = prices['volume'].values
    
    # Calculate weekly pivot points (using previous week's OHLC)
    # We'll use weekly resampled data from daily data to avoid look-ahead
    # Since we can't resample, we'll approximate using daily data and calculate weekly pivot from prior week
    # For simplicity, we'll use daily high/low/close to calculate weekly pivot (standard practice)
    # But to avoid look-ahead, we need to use only completed weekly data
    # Instead, we'll use daily data and calculate pivot for the current week using prior day's data? 
    # Actually, standard weekly pivot uses prior week's OHLC
    # We'll approximate by using daily data and shifting to get prior week's values
    
    # Convert to DataFrame for easier resampling (but we'll do it manually to avoid look-ahead)
    # Since we cannot resample in the loop, we'll calculate weekly pivot using daily data with proper shift
    
    # Let's use a simpler approach: calculate daily pivot and use it as weekly approximation
    # But better: use 1d data to get prior week's OHLC by looking back 5 days
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly high, low, close from prior 5 trading days (approximately 1 week)
    # We'll use rolling window of 5 days on daily data
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Weekly high = max of prior 5 days
    weekly_high = pd.Series(daily_high).rolling(window=5, min_periods=5).max().shift(1).values  # shift to avoid look-ahead
    weekly_low = pd.Series(daily_low).rolling(window=5, min_periods=5).min().shift(1).values
    weekly_close = pd.Series(daily_close).rolling(window=5, min_periods=5).last().shift(1).values
    
    # Calculate pivot points
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    r2 = pp + (weekly_high - weekly_low)
    s2 = pp - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pp - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pp)
    r4 = weekly_high + 3 * (weekly_high - weekly_low)
    s4 = weekly_low - 3 * (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 12h trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_12h_up = close_12h > ema50_12h
    trend_12h_down = close_12h < ema50_12h
    
    # Align 12h trend to 6h
    trend_12h_up_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_up.astype(float))
    trend_12h_down_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_down.astype(float))
    
    # Volume filter: volume > 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(trend_12h_up_aligned[i]) or np.isnan(trend_12h_down_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Fade at R3/S3: price touches R3 and reverses down, or touches S3 and reverses up
            # Breakout at R4/S4: price breaks above R4 or below S4 with volume
            
            # Fade short at R3: price crosses below R3 from above
            if (close[i] < r3_aligned[i] and 
                close[i-1] >= r3_aligned[i-1] and
                trend_12h_down_aligned[i] > 0.5 and
                volume_filter[i]):
                signals[i] = -0.25
                position = -1
            # Fade long at S3: price crosses above S3 from below
            elif (close[i] > s3_aligned[i] and 
                  close[i-1] <= s3_aligned[i-1] and
                  trend_12h_up_aligned[i] > 0.5 and
                  volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Breakout long at R4: price crosses above R4
            elif (close[i] > r4_aligned[i] and 
                  close[i-1] <= r4_aligned[i-1] and
                  trend_12h_up_aligned[i] > 0.5 and
                  volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Breakout short at S4: price crosses below S4
            elif (close[i] < s4_aligned[i] and 
                  close[i-1] >= s4_aligned[i-1] and
                  trend_12h_down_aligned[i] > 0.5 and
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit conditions: price returns to pivot, or trend reverses, or touches S1/S2
            if (close[i] < pp_aligned[i] or
                trend_12h_up_aligned[i] < 0.5 or
                close[i] < s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: price returns to pivot, or trend reverses, or touches R1/R2
            if (close[i] > pp_aligned[i] or
                trend_12h_down_aligned[i] < 0.5 or
                close[i] > r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals