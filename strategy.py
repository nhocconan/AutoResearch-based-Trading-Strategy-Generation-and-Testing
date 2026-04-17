# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian_Breakout_V2
Hypothesis: Use weekly pivot levels (R1/S1) as key support/resistance on 6h timeframe.
Breakouts above R1 with volume confirmation signal long positions; breakdowns below S1 with volume confirmation signal short positions.
Uses Donchian channel (20) on 1d to filter breakouts in trending markets only. Designed for low trade frequency (15-30/year) and works in both bull/bear markets by requiring breakout confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot points: P, R1, S1, R2, S2"""
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, s1, r2, s2

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Weekly data for pivot points ===
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivots
    _, weekly_r1, weekly_s1, _, _ = calculate_weekly_pivot(weekly_high, weekly_low, weekly_close)
    
    # Align weekly pivot levels to 6h
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    
    # === Daily Donchian channel for trend filter ===
    df_daily = get_htf_data(prices, '1d')
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    
    # Donchian channel (20)
    dc_high = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    dc_low = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    
    dc_high_aligned = align_htf_to_ltf(prices, df_daily, dc_high)
    dc_low_aligned = align_htf_to_ltf(prices, df_daily, dc_low)
    
    # === Volume filter ===
    # Daily volume average (20)
    daily_volume = df_daily['volume'].values
    vol_avg_daily = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    vol_avg_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_daily)
    
    # Current daily volume for confirmation
    daily_volume_aligned = align_htf_to_ltf(prices, df_daily, daily_volume)
    
    signals = np.zeros(n)
    
    # Warmup: need enough data for weekly pivot and daily Donchian
    warmup = 20  # Donchian period
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or
            np.isnan(dc_high_aligned[i]) or
            np.isnan(dc_low_aligned[i]) or
            np.isnan(vol_avg_daily_aligned[i]) or
            np.isnan(daily_volume_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume filter: current daily volume > 1.3x average
        vol_filter = daily_volume_aligned[i] > 1.3 * vol_avg_daily_aligned[i]
        
        # Breakout conditions
        # Long: price breaks above weekly R1 AND above daily Donchian high (trend confirmation)
        long_breakout = (close[i] > weekly_r1_aligned[i]) and (close[i] > dc_high_aligned[i])
        
        # Short: price breaks below weekly S1 AND below daily Donchian low (trend confirmation)
        short_breakout = (close[i] < weekly_s1_aligned[i]) and (close[i] < dc_low_aligned[i])
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: breakout above R1 with volume and trend confirmation
            if long_breakout and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: breakdown below S1 with volume and trend confirmation
            elif short_breakout and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit when price closes below weekly pivot OR below Donchian low
            weekly_pivot = (weekly_r1_aligned[i] + weekly_s1_aligned[i]) / 2  # approximate pivot
            if (close[i] < weekly_pivot) or (close[i] < dc_low_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price closes above weekly pivot OR above Donchian high
            weekly_pivot = (weekly_r1_aligned[i] + weekly_s1_aligned[i]) / 2  # approximate pivot
            if (close[i] > weekly_pivot) or (close[i] > dc_high_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Donchian_Breakout_V2"
timeframe = "6h"
leverage = 1.0