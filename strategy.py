#!/usr/bin/env python3
"""
6h_1d_WeeklyPivot_DonchianBreakout
Hypothesis: Use weekly pivot levels to determine market regime and 1d Donchian breakouts for entries on 6h.
Long when price breaks above 1d Donchian high (20) AND weekly pivot > prior weekly pivot (bullish regime).
Short when price breaks below 1d Donchian low (20) AND weekly pivot < prior weekly pivot (bearish regime).
Exit when price returns to 1d Donchian midpoint.
Uses weekly regime to avoid counter-trend trades and Donchian for clear breakout signals.
Designed for 6h to limit trade frequency (target: 15-35/year) and reduce fee drift.
Works in bull markets by buying breakouts in uptrend and in bear markets by selling breakdowns in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for regime (pivot points)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot: (H + L + C) / 3
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    
    # Align weekly pivot to 6h
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    
    # Load daily data for Donchian channels
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    
    # Calculate Donchian channels (20-period)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.nanmax(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.nanmin(arr[i-window+1:i+1])
        return res
    
    donch_high = rolling_max(daily_high, 20)
    donch_low = rolling_min(daily_low, 20)
    donch_mid = (donch_high + donch_low) / 2
    
    # Align Donchian levels to 6h
    donch_high_aligned = align_htf_to_ltf(prices, df_daily, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_daily, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_daily, donch_mid)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(donch_mid_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        # Determine weekly regime: compare current vs previous weekly pivot
        # Need to get the weekly pivot value from the previous week
        week_idx = i // (7 * 24 * 60 // 6)  # Approximate weekly index in 6h bars
        if week_idx < 1:
            continue
            
        # Get previous weekly pivot (simplified: use prior value)
        # We'll use the fact that weekly_pivot_aligned holds the value for the completed week
        # For regime, we need to know if weekly pivot is rising or falling
        if i >= 6:  # At least one 6h bar into the week
            # Simple trend: current weekly pivot vs value 6 periods ago (approx one week)
            if not np.isnan(weekly_pivot_aligned[i]) and not np.isnan(weekly_pivot_aligned[i-6]):
                weekly_rising = weekly_pivot_aligned[i] > weekly_pivot_aligned[i-6]
                weekly_falling = weekly_pivot_aligned[i] < weekly_pivot_aligned[i-6]
            else:
                weekly_rising = False
                weekly_falling = False
        else:
            weekly_rising = False
            weekly_falling = False
        
        if position == 0:
            # Long: Donchian breakout AND bullish weekly regime
            if price > donch_high_aligned[i] and weekly_rising:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown AND bearish weekly regime
            elif price < donch_low_aligned[i] and weekly_falling:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to Donchian midpoint
            if price <= donch_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to Donchian midpoint
            if price >= donch_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_WeeklyPivot_DonchianBreakout"
timeframe = "6h"
leverage = 1.0