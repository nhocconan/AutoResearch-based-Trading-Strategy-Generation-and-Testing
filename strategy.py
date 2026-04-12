#!/usr/bin/env python3
"""
6h_1w_1d_RangeBreakout_v1
Hypothesis: Use weekly range to define trend bias and daily range for entry timing.
In bull market (price > weekly midpoint), buy dips to daily support with volume confirmation.
In bear market (price < weekly midpoint), sell rallies to daily resistance with volume confirmation.
Uses 6h timeframe for lower frequency (~20-40 trades/year) to reduce fee drag.
Works in bull via buying weakness, in bear via selling strength.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_RangeBreakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's OHLC for range
    prev_week_high = df_1w['high'].iloc[-2] if len(df_1w) >= 2 else df_1w['high'].iloc[-1]
    prev_week_low = df_1w['low'].iloc[-2] if len(df_1w) >= 2 else df_1w['low'].iloc[-1]
    prev_week_close = df_1w['close'].iloc[-2] if len(df_1w) >= 2 else df_1w['close'].iloc[-1]
    
    # Calculate weekly midpoint
    weekly_midpoint = (prev_week_high + prev_week_low) / 2
    weekly_midpoint_array = np.full(len(df_1w), weekly_midpoint)
    weekly_midpoint_aligned = align_htf_to_ltf(prices, df_1w, weekly_midpoint_array)
    
    # Daily data for entry levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for support/resistance
    prev_day_high = df_1d['high'].iloc[-2] if len(df_1d) >= 2 else df_1d['high'].iloc[-1]
    prev_day_low = df_1d['low'].iloc[-2] if len(df_1d) >= 2 else df_1d['low'].iloc[-1]
    
    # Daily support and resistance (simple)
    daily_support = prev_day_low
    daily_resistance = prev_day_high
    
    # Align daily levels to 6h timeframe
    daily_support_array = np.full(len(df_1d), daily_support)
    daily_resistance_array = np.full(len(df_1d), daily_resistance)
    daily_support_aligned = align_htf_to_ltf(prices, df_1d, daily_support_array)
    daily_resistance_aligned = align_htf_to_ltf(prices, df_1d, daily_resistance_array)
    
    # Volume average (24-period for 6h ~ 4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any data invalid
        if (np.isnan(weekly_midpoint_aligned[i]) or
            np.isnan(daily_support_aligned[i]) or np.isnan(daily_resistance_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend bias from weekly midpoint
        bullish_bias = close[i] > weekly_midpoint_aligned[i]
        bearish_bias = close[i] < weekly_midpoint_aligned[i]
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma[i]
        
        # Entry conditions
        long_entry = bullish_bias and close[i] <= daily_support_aligned[i] and vol_spike
        short_entry = bearish_bias and close[i] >= daily_resistance_aligned[i] and vol_spike
        
        # Exit conditions: opposite touch
        long_exit = close[i] >= daily_resistance_aligned[i]
        short_exit = close[i] <= daily_support_aligned[i]
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals