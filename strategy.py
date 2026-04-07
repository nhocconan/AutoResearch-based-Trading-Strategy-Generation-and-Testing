#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Weekly Pivot Range Breakout with Volume Confirmation
# Hypothesis: Price breaking above weekly R4 or below S4 with volume surge continues in breakout direction.
# Uses weekly pivot levels for structural support/resistance and volume to filter false breakouts.
# Works in bull/bear markets by trading breakouts in direction of weekly trend.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_weekly_pivot_range_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's data)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Pivot point calculations
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    r4 = weekly_high + 3 * (pivot - weekly_low)
    s4 = weekly_low - 3 * (weekly_high - pivot)
    
    # Align weekly pivot levels to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r4_aligned = align_htf_to_ltf(prices, df_weekly, r4)
    s4_aligned = align_htf_to_ltf(prices, df_weekly, s4)
    
    # Volume moving average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(200, n):
        # Skip if required data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(volume_ma[i]):
            continue
        
        # Volume surge condition (current volume > 1.5x average)
        volume_surge = volume[i] > 1.5 * volume_ma[i]
        
        # Breakout conditions
        bullish_breakout = close[i] > r4_aligned[i] and volume_surge
        bearish_breakout = close[i] < s4_aligned[i] and volume_surge
        
        if bullish_breakout:
            signals[i] = 0.25
        elif bearish_breakout:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals