#!/usr/bin/env python3
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
    
    # Get 12h data for higher timeframe context (primary HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get 1d data for weekly pivot context (secondary HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 12h EMA 50 for trend direction
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 6h Donchian channels (20-period for structure)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.8x 30-period average (strong filter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    # Weekly pivot levels from 1d data (using daily high/low/close)
    # Calculate pivot points for each day
    pivot_points = (high_1d + low_1d + close_1d) / 3
    # Calculate weekly pivot levels (using last 5 days of daily data)
    # We'll use the most recent weekly pivot based on 5-day lookback
    weekly_pivot_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_pivot_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_pivot_close = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Calculate weekly pivot points (standard formula)
    weekly_pivot = (weekly_pivot_high + weekly_pivot_low + weekly_pivot_close) / 3
    weekly_r1 = 2 * weekly_pivot - weekly_pivot_low
    weekly_s1 = 2 * weekly_pivot - weekly_pivot_high
    weekly_r2 = weekly_pivot + (weekly_pivot_high - weekly_pivot_low)
    weekly_s2 = weekly_pivot - (weekly_pivot_high - weekly_pivot_low)
    weekly_r3 = weekly_pivot_high + 2 * (weekly_pivot - weekly_pivot_low)
    weekly_s3 = weekly_pivot_low - 2 * (weekly_pivot_high - weekly_pivot)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(weekly_r3_aligned[i]) or np.isnan(weekly_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA50
        price_above_ema = close[i] > ema_50_12h_aligned[i]
        price_below_ema = close[i] < ema_50_12h_aligned[i]
        
        # Long conditions: price breaks above upper Donchian + above 12h EMA + volume + above weekly R3
        long_breakout = (close[i] > highest_high[i-1] and price_above_ema and volume_filter[i] and close[i] > weekly_r3_aligned[i])
        # Short conditions: price breaks below lower Donchian + below 12h EMA + volume + below weekly S3
        short_breakout = (close[i] < lowest_low[i-1] and price_below_ema and volume_filter[i] and close[i] < weekly_s3_aligned[i])
        
        if long_breakout:
            signals[i] = 0.25
            position = 1
        elif short_breakout:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite Donchian breakout
        elif position == 1 and close[i] < lowest_low[i-1]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > highest_high[i-1]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Donchian20_Breakout_12hEMA50_WeeklyPivotR3S3_VolumeFilter"
timeframe = "6h"
leverage = 1.0