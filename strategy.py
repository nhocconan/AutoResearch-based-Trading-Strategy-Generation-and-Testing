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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot using previous day (yesterday's data)
    # Use yesterday's high, low, close for today's pivot
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev = np.roll(close_1d, 1)
    
    pivot_daily = (high_prev + low_prev + close_prev) / 3.0
    range_prev = high_prev - low_prev
    r2_daily = pivot_daily + range_prev  # R2 = Pivot + Range
    s2_daily = pivot_daily - range_prev  # S2 = Pivot - Range
    
    # Calculate daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 4h timeframe
    r2_daily_aligned = align_htf_to_ltf(prices, df_1d, r2_daily)
    s2_daily_aligned = align_htf_to_ltf(prices, df_1d, s2_daily)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    pivot_daily_aligned = align_htf_to_ltf(prices, df_1d, pivot_daily)
    
    # Calculate volume ratio (current vs 20-period average)
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
        if (np.isnan(r2_daily_aligned[i]) or 
            np.isnan(s2_daily_aligned[i]) or
            np.isnan(ema50_aligned[i]) or
            np.isnan(vol_ratio[i]) or
            np.isnan(pivot_daily_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA50
        uptrend = close[i] > ema50_aligned[i]
        downtrend = close[i] < ema50_aligned[i]
        
        # Volume filter: current volume above 2.0x average (stricter for fewer trades)
        vol_filter = vol_ratio[i] > 2.0
        
        # Breakout conditions: price breaks daily R2/S2 with volume and trend
        long_breakout = close[i] > r2_daily_aligned[i]
        short_breakout = close[i] < s2_daily_aligned[i]
        
        long_entry = long_breakout and uptrend and vol_filter
        short_entry = short_breakout and downtrend and vol_filter
        
        # Exit conditions: price returns to daily pivot level or trend reverses
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

name = "4h_DailyPivot_R2S2_Breakout_1dEMA50_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0