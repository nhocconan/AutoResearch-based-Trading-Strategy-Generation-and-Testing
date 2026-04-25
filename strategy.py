#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_WeeklyPivotDir_v1
Hypothesis: Trade 6h Camarilla R3/S3 breakouts aligned with 1d trend and weekly pivot direction.
- Use 1d EMA50 for trend filter (bullish: price > EMA50, bearish: price < EMA50)
- Use weekly pivot (from prior week) for directional bias (bullish: price > weekly pivot, bearish: price < weekly pivot)
- Only take long when both 1d trend and weekly pivot are bullish and price breaks above R3
- Only take short when both 1d trend and weekly pivot are bearish and price breaks below S3
- Volume confirmation: require volume > 1.8x 20-period average to filter weak breakouts
- Position size: 0.25. Target: 50-150 total trades over 4 years = 12-37/year.
- Works in bull markets via trend following and in bear markets via short-side symmetry.
- Weekly pivot adds structural bias from higher timeframe, reducing counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1w data for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot from prior week's OHLC
    prev_weekly_close = np.roll(df_1w['close'].values, 1)
    prev_weekly_high = np.roll(df_1w['high'].values, 1)
    prev_weekly_low = np.roll(df_1w['low'].values, 1)
    # Handle first bar
    prev_weekly_close[0] = df_1w['close'].values[0]
    prev_weekly_high[0] = df_1w['high'].values[0]
    prev_weekly_low[0] = df_1w['low'].values[0]
    
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Get daily data for Camarilla levels (using prior day's OHLC)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close[0] = close_1d[0]
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    
    daily_pivot = (prev_high + prev_low + prev_close) / 3.0
    daily_range = prev_high - prev_low
    
    # Camarilla levels (R3/S3 are the stronger breakout levels)
    r3 = daily_pivot + (daily_range * 1.1 / 4)
    s3 = daily_pivot - (daily_range * 1.1 / 4)
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    daily_pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot)
    
    # Volume spike confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(daily_pivot_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d trend (bullish/bearish)
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Determine weekly pivot bias
        weekly_bullish = close[i] > weekly_pivot_aligned[i]
        weekly_bearish = close[i] < weekly_pivot_aligned[i]
        
        if position == 0:
            # Long setup: bullish 1d trend + bullish weekly pivot + R3 breakout + volume spike
            long_setup = (htf_1d_bullish and weekly_bullish and 
                         close[i] > r3_aligned[i] and volume_spike[i])
            
            # Short setup: bearish 1d trend + bearish weekly pivot + S3 breakdown + volume spike
            short_setup = (htf_1d_bearish and weekly_bearish and 
                          close[i] < s3_aligned[i] and volume_spike[i])
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: trend reversal, weekly bias change, or mean reversion to daily pivot
            exit_signal = (not htf_1d_bullish) or (not weekly_bullish) or (close[i] < daily_pivot_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: trend reversal, weekly bias change, or mean reversion to daily pivot
            exit_signal = htf_1d_bullish or weekly_bullish or (close[i] > daily_pivot_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_WeeklyPivotDir_v1"
timeframe = "6h"
leverage = 1.0