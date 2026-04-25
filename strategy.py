#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Spike
Hypothesis: Donchian breakouts capture momentum. Weekly pivot direction (from 1d data) provides higher-timeframe bias to filter breakouts. Volume spike confirms institutional participation. Designed for 6h timeframe to avoid overtrading while working in both bull and bear markets via directional filter.
"""

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
    
    # Get 1d data for weekly pivot calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week's OHLC
    # Need at least 5 days to form a complete week (Mon-Fri)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly high/low/close: use last 5 trading days (approximation)
    # For simplicity, use rolling window of 5 days on 1d data
    week_high = np.full(len(high_1d), np.nan)
    week_low = np.full(len(low_1d), np.nan)
    week_close = np.full(len(close_1d), np.nan)
    
    for i in range(4, len(high_1d)):  # need 5 days: i-4 to i
        week_high[i] = np.max(high_1d[i-4:i+1])
        week_low[i] = np.min(low_1d[i-4:i+1])
        week_close[i] = close_1d[i]  # Friday's close
    
    # Weekly pivot calculations (based on prior week)
    pivot_point = (week_high + week_low + week_close) / 3.0
    r1 = 2 * pivot_point - week_low
    s1 = 2 * pivot_point - week_high
    r2 = pivot_point + (week_high - week_low)
    s2 = pivot_point - (week_high - week_low)
    r3 = week_high + 2 * (pivot_point - week_low)
    s3 = week_low - 2 * (week_high - pivot_point)
    
    # Align weekly pivot levels to 6h timeframe (completed weekly bar only)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Get 6h data for Donchian channel (primary timeframe)
    # Donchian(20): 20-period high/low
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(19, n):  # 20 periods: i-19 to i
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Calculate ATR(14) for stop loss
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for Donchian, ATR, volume MA, and weekly pivot
    start_idx = max(19, 14, 20, 4)  # weekly pivot needs 4+ for 5-day lookback
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Weekly pivot direction: price above/below weekly pivot
        # Weekly pivot is based on completed weekly bar, so aligned array gives prior week's pivot
        above_weekly_pivot = curr_close > pivot_aligned[i]
        below_weekly_pivot = curr_close < pivot_aligned[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long breakout: price breaks above Donchian(20) high with volume confirmation and above weekly pivot
            long_breakout = (curr_close > donchian_high[i]) and volume_confirm and above_weekly_pivot
            # Short breakout: price breaks below Donchian(20) low with volume confirmation and below weekly pivot
            short_breakout = (curr_close < donchian_low[i]) and volume_confirm and below_weekly_pivot
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit conditions: price closes below Donchian(20) low OR 2*ATR trailing stop
            if curr_close < donchian_low[i] or curr_close < (highest_since_entry - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit conditions: price closes above Donchian(20) high OR 2*ATR trailing stop
            if curr_close > donchian_high[i] or curr_close > (lowest_since_entry + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_Breakout_WeeklyPivot_Direction_VolumeSpike"
timeframe = "6h"
leverage = 1.0