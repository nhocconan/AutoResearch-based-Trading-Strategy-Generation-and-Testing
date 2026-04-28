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
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot (classic)
    pivot_daily = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1_daily = pivot_daily + (range_1d * 1.0)
    s1_daily = pivot_daily - (range_1d * 1.0)
    
    # Calculate daily EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align daily indicators to 4h timeframe
    r1_daily_aligned = align_htf_to_ltf(prices, df_1d, r1_daily)
    s1_daily_aligned = align_htf_to_ltf(prices, df_1d, s1_daily)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate volume filter (24-period MA on 4h = 4 days)
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
        if (np.isnan(r1_daily_aligned[i]) or 
            np.isnan(s1_daily_aligned[i]) or
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
        
        # Breakout conditions: price breaks daily R1/S1 with volume and trend
        long_breakout = close[i] > r1_daily_aligned[i]
        short_breakout = close[i] < s1_daily_aligned[i]
        
        long_entry = long_breakout and uptrend and vol_filter
        short_entry = short_breakout and downtrend and vol_filter
        
        # Exit conditions: price returns to daily pivot level or trend reverses
        long_exit = close[i] < pivot_daily[i] or not uptrend
        short_exit = close[i] > pivot_daily[i] or not downtrend
        
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

name = "4h_DailyPivot_R1S1_Breakout_1dEMA200_Volume_v2"
timeframe = "4h"
leverage = 1.0