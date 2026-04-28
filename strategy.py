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
    
    # Get weekly data for Donchian and pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channel (20 weeks)
    high_20w = pd.Series(high_1w).rolling(window=20, min_periods=20).max()
    low_20w = pd.Series(low_1w).rolling(window=20, min_periods=20).min()
    
    # Calculate weekly pivot points (standard)
    pivot_weekly = (high_1w + low_1w + close_1w) / 3.0
    range_weekly = high_1w - low_1w
    r1_weekly = pivot_weekly + range_weekly
    s1_weekly = pivot_weekly - range_weekly
    r2_weekly = pivot_weekly + 2 * range_weekly
    s2_weekly = pivot_weekly - 2 * range_weekly
    
    # Align weekly indicators to 6h timeframe
    high_20w_aligned = align_htf_to_ltf(prices, df_1w, high_20w)
    low_20w_aligned = align_htf_to_ltf(prices, df_1w, low_20w)
    pivot_weekly_aligned = align_htf_to_ltf(prices, df_1w, pivot_weekly)
    r1_weekly_aligned = align_htf_to_ltf(prices, df_1w, r1_weekly)
    s1_weekly_aligned = align_htf_to_ltf(prices, df_1w, s1_weekly)
    r2_weekly_aligned = align_htf_to_ltf(prices, df_1w, r2_weekly)
    s2_weekly_aligned = align_htf_to_ltf(prices, df_1w, s2_weekly)
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: 6-period average volume (1 day on 6h)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20w_aligned[i]) or 
            np.isnan(low_20w_aligned[i]) or
            np.isnan(pivot_weekly_aligned[i]) or
            np.isnan(r1_weekly_aligned[i]) or
            np.isnan(s1_weekly_aligned[i]) or
            np.isnan(r2_weekly_aligned[i]) or
            np.isnan(s2_weekly_aligned[i]) or
            np.isnan(ema50_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA50
        uptrend = close[i] > ema50_aligned[i]
        downtrend = close[i] < ema50_aligned[i]
        
        # Volume filter: current volume above average
        vol_filter = volume[i] > vol_ma[i]
        
        # Breakout conditions: price breaks weekly Donchian levels
        long_breakout = close[i] > high_20w_aligned[i]
        short_breakout = close[i] < low_20w_aligned[i]
        
        # Entry conditions: breakout with trend and volume confirmation
        long_entry = long_breakout and uptrend and vol_filter
        short_entry = short_breakout and downtrend and vol_filter
        
        # Exit conditions: price returns to weekly pivot or opposite S1/R1 level
        long_exit = close[i] < pivot_weekly_aligned[i] or close[i] < s1_weekly_aligned[i]
        short_exit = close[i] > pivot_weekly_aligned[i] or close[i] > r1_weekly_aligned[i]
        
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

name = "6h_Donchian20_WeeklyPivot_Trend_Volume"
timeframe = "6h"
leverage = 1.0