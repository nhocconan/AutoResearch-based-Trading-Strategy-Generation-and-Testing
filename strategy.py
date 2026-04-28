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
    
    # Get weekly data for monthly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate monthly pivot using last 4 weeks
    high_4w = pd.Series(high_1w).rolling(window=4, min_periods=4).max()
    low_4w = pd.Series(low_1w).rolling(window=4, min_periods=4).min()
    close_4w = pd.Series(close_1w).rolling(window=4, min_periods=4).last()
    
    pivot_monthly = (high_4w + low_4w + close_4w) / 3.0
    range_4w = high_4w - low_4w
    r3_monthly = pivot_monthly + (range_4w * 1.1 / 2.0)
    s3_monthly = pivot_monthly - (range_4w * 1.1 / 2.0)
    
    # Calculate monthly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align monthly indicators to 12h timeframe
    r3_monthly_aligned = align_htf_to_ltf(prices, df_1w, r3_monthly)
    s3_monthly_aligned = align_htf_to_ltf(prices, df_1w, s3_monthly)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate average volume over 4 periods (2 days on 12h)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_monthly_aligned[i]) or 
            np.isnan(s3_monthly_aligned[i]) or
            np.isnan(ema50_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA50
        uptrend = close[i] > ema50_aligned[i]
        downtrend = close[i] < ema50_aligned[i]
        
        # Volume filter: current volume above average
        vol_filter = volume[i] > vol_ma[i]
        
        # Breakout conditions: price breaks monthly R3/S3 with volume and trend
        long_breakout = close[i] > r3_monthly_aligned[i]
        short_breakout = close[i] < s3_monthly_aligned[i]
        
        long_entry = long_breakout and uptrend and vol_filter
        short_entry = short_breakout and downtrend and vol_filter
        
        # Exit conditions: price returns to monthly pivot level or trend reverses
        pivot_monthly_series = pd.Series(pivot_monthly)
        pivot_monthly_last = pivot_monthly_series.rolling(window=4, min_periods=4).last().values
        pivot_monthly_aligned = align_htf_to_ltf(prices, df_1w, pivot_monthly_last)
        long_exit = close[i] < pivot_monthly_aligned[i] or not uptrend
        short_exit = close[i] > pivot_monthly_aligned[i] or not downtrend
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
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
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_MonthlyPivot_R3S3_Breakout_1wEMA50_Volume_v1"
timeframe = "12h"
leverage = 1.0