#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Weekly EMA(20) for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Daily Donchian(20) channels
    high_max = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align weekly and daily indicators to daily timeframe
    ema20_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    high_max_aligned = align_htf_to_ltf(prices, df_1d, high_max)
    low_min_aligned = align_htf_to_ltf(prices, df_1d, low_min)
    
    # Calculate average volume over 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_aligned[i]) or 
            np.isnan(high_max_aligned[i]) or
            np.isnan(low_min_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA20
        uptrend = close[i] > ema20_aligned[i]
        downtrend = close[i] < ema20_aligned[i]
        
        # Volume filter: current volume above average
        vol_filter = volume[i] > vol_ma[i]
        
        # Breakout conditions: price breaks Donchian channels with volume and trend
        long_breakout = close[i] > high_max_aligned[i]
        short_breakout = close[i] < low_min_aligned[i]
        
        long_entry = long_breakout and uptrend and vol_filter
        short_entry = short_breakout and downtrend and vol_filter
        
        # Exit conditions: price returns to opposite Donchian level or trend reverses
        long_exit = close[i] < low_min_aligned[i] or not uptrend
        short_exit = close[i] > high_max_aligned[i] or not downtrend
        
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

name = "1d_WeeklyTrend_Donchian20_Volume"
timeframe = "1d"
leverage = 1.0