#!/usr/bin/env python3
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
    
    # Get daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot using last 5 days
    high_5d = pd.Series(high_1d).rolling(window=5, min_periods=5).max()
    low_5d = pd.Series(low_1d).rolling(window=5, min_periods=5).min()
    close_5d = pd.Series(close_1d).rolling(window=5, min_periods=5).last()
    
    pivot_weekly = (high_5d + low_5d + close_5d) / 3.0
    range_5d = high_5d - low_5d
    r3_weekly = pivot_weekly + (range_5d * 1.1 / 2.0)
    s3_weekly = pivot_weekly - (range_5d * 1.1 / 2.0)
    
    # Calculate weekly EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly indicators to 1h timeframe
    r3_weekly_aligned = align_htf_to_ltf(prices, df_1d, r3_weekly)
    s3_weekly_aligned = align_htf_to_ltf(prices, df_1d, s3_weekly)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate average volume over 24 periods (1 day on 1h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_weekly_aligned[i]) or 
            np.isnan(s3_weekly_aligned[i]) or
            np.isnan(ema200_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA200
        uptrend = close[i] > ema200_aligned[i]
        downtrend = close[i] < ema200_aligned[i]
        
        # Volume filter: current volume above average
        vol_filter = volume[i] > vol_ma[i]
        
        # Breakout conditions: price breaks weekly R3/S3 with volume and trend
        long_breakout = close[i] > r3_weekly_aligned[i]
        short_breakout = close[i] < s3_weekly_aligned[i]
        
        long_entry = long_breakout and uptrend and vol_filter
        short_entry = short_breakout and downtrend and vol_filter
        
        # Exit conditions: price returns to weekly pivot level or trend reverses
        pivot_weekly_series = pd.Series(pivot_weekly)
        pivot_weekly_last = pivot_weekly_series.rolling(window=5, min_periods=5).last().values
        pivot_weekly_aligned = align_htf_to_ltf(prices, df_1d, pivot_weekly_last)
        long_exit = close[i] < pivot_weekly_aligned[i] or not uptrend
        short_exit = close[i] > pivot_weekly_aligned[i] or not downtrend
        
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_WeeklyPivot_R3S3_Breakout_1dEMA200_Volume_v3"
timeframe = "1h"
leverage = 1.0