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
    
    # Get weekly data for pivot calculation (weekly pivot from previous week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot (from previous week's OHLC)
    pivot_weekly = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r1_weekly = pivot_weekly + (range_1w * 0.382)  # R1 = Pivot + 0.382 * Range
    s1_weekly = pivot_weekly - (range_1w * 0.382)  # S1 = Pivot - 0.382 * Range
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to daily timeframe
    r1_weekly_aligned = align_htf_to_ltf(prices, df_1w, r1_weekly)
    s1_weekly_aligned = align_htf_to_ltf(prices, df_1w, s1_weekly)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma20 + 1e-10)
    
    # Precompute session filter (08-20 UTC) - trade only during active hours
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_weekly_aligned[i]) or 
            np.isnan(s1_weekly_aligned[i]) or
            np.isnan(ema50_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA50
        uptrend = close[i] > ema50_aligned[i]
        downtrend = close[i] < ema50_aligned[i]
        
        # Volume filter: current volume above 1.5x average
        vol_filter = vol_ratio[i] > 1.5
        
        # Breakout conditions: price breaks weekly R1/S1 with volume and trend
        long_breakout = close[i] > r1_weekly_aligned[i]
        short_breakout = close[i] < s1_weekly_aligned[i]
        
        long_entry = long_breakout and uptrend and vol_filter
        short_entry = short_breakout and downtrend and vol_filter
        
        # Exit conditions: price returns to weekly pivot level or trend reverses
        pivot_weekly_aligned = align_htf_to_ltf(prices, df_1w, pivot_weekly)
        long_exit = close[i] < pivot_weekly_aligned[i] or not uptrend
        short_exit = close[i] > pivot_weekly_aligned[i] or not downtrend
        
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

name = "1d_WeeklyPivot_R1S1_Breakout_1wEMA50_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0