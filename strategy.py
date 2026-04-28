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
    
    # Get daily data once for HTF context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for additional context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily high/low/close for calculations
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Weekly high/low/close for calculations
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate daily range for pivot calculations
    daily_range = high_1d - low_1d
    
    # Pivot point (classic)
    pivot = (high_1d + low_1d + close_1d) / 3
    
    # Calculate weekly high-low range
    weekly_range = high_1w - low_1w
    
    # Align daily pivot and weekly range to daily timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    weekly_range_aligned = align_htf_to_ltf(prices, df_1w, weekly_range)
    
    # Align daily data
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # Daily range aligned
    daily_range_aligned = high_1d_aligned - low_1d_aligned
    
    # Calculate resistance and support levels (R1, S1)
    # R1 = 2*P - L, S1 = 2*P - H
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Align R1 and S1
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: above average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(weekly_range_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(high_1d_aligned[i]) or
            np.isnan(low_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: above average volume
        vol_filter = volume[i] > vol_ma[i]
        
        # Weekly trend filter: price above/below weekly midpoint
        weekly_midpoint = low_1w + weekly_range_aligned[i] / 2
        low_1w_aligned = align_htf_to_ltf(prices, df_1w, low_1w)
        weekly_midpoint = low_1w_aligned[i] + weekly_range_aligned[i] / 2
        
        price_above_weekly_mid = close[i] > weekly_midpoint
        price_below_weekly_mid = close[i] < weekly_midpoint
        
        # Price action: break above R1 or below S1
        price_above_r1 = close[i] > r1_aligned[i]
        price_below_s1 = close[i] < s1_aligned[i]
        
        # Entry conditions: 
        # Long: price breaks above R1 with volume and above weekly midpoint
        # Short: price breaks below S1 with volume and below weekly midpoint
        long_entry = price_above_r1 and price_above_weekly_mid and vol_filter
        short_entry = price_below_s1 and price_below_weekly_mid and vol_filter
        
        # Exit conditions: price returns to pivot level
        long_exit = close[i] < pivot_aligned[i]
        short_exit = close[i] > pivot_aligned[i]
        
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
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_WeeklyTrend"
timeframe = "1d"
leverage = 1.0