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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Get daily data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily high/low/close for pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot using previous day (yesterday)
    high_1d_prev = np.concatenate([high_1d[0:1], high_1d[:-1]])
    low_1d_prev = np.concatenate([low_1d[0:1], low_1d[:-1]])
    close_1d_prev = np.concatenate([close_1d[0:1], close_1d[:-1]])
    
    pivot_daily = (high_1d_prev + low_1d_prev + close_1d_prev) / 3.0
    range_1d = high_1d_prev - low_1d_prev
    r3_daily = pivot_daily + (range_1d * 1.1)  # R3 = Pivot + 1.1 * Range
    s3_daily = pivot_daily - (range_1d * 1.1)  # S3 = Pivot - 1.1 * Range
    
    # Align weekly and daily indicators to 12h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    r3_daily_aligned = align_htf_to_ltf(prices, df_1d, r3_daily)
    s3_daily_aligned = align_htf_to_ltf(prices, df_1d, s3_daily)
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma20 + 1e-10)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(r3_daily_aligned[i]) or
            np.isnan(s3_daily_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA50
        uptrend = close[i] > ema50_1w_aligned[i]
        downtrend = close[i] < ema50_1w_aligned[i]
        
        # Volume filter: current volume above 2.0x average
        vol_filter = vol_ratio[i] > 2.0
        
        # Breakout conditions: price breaks daily R3/S3 with volume and trend
        long_breakout = close[i] > r3_daily_aligned[i]
        short_breakout = close[i] < s3_daily_aligned[i]
        
        long_entry = long_breakout and uptrend and vol_filter
        short_entry = short_breakout and downtrend and vol_filter
        
        # Exit conditions: price returns to daily pivot level or trend reverses
        pivot_daily_aligned = align_htf_to_ltf(prices, df_1d, pivot_daily)
        long_exit = close[i] < pivot_daily_aligned[i] or not uptrend
        short_exit = close[i] > pivot_daily_aligned[i] or not downtrend
        
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

name = "12h_WeeklyEMA50_Trend_DailyPivot_R3S3_Breakout_v1"
timeframe = "12h"
leverage = 1.0