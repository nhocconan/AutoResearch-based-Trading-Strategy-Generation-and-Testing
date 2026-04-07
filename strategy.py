#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Weekly Pivot Range Breakout + Volume
# Hypothesis: Breakouts beyond weekly pivot R4/S4 levels with volume confirmation
# capture strong momentum moves. Works in both bull and bear by trading breakouts
# in either direction. Weekly pivots provide robust support/resistance levels.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_weekly_pivot_range_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points
    close_weekly = df_weekly['close'].values
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Previous week's data for pivot calculation
    prev_close = np.roll(close_weekly, 1)
    prev_high = np.roll(high_weekly, 1)
    prev_low = np.roll(low_weekly, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Weekly pivot point and support/resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    width = prev_high - prev_low
    
    # Weekly R4 and S4 (strong breakout levels)
    R4 = pivot + width * 1.1
    S4 = pivot - width * 1.1
    
    # Align weekly levels to 6h
    R4_6h = align_htf_to_ltf(prices, df_weekly, R4)
    S4_6h = align_htf_to_ltf(prices, df_weekly, S4)
    
    # Volume filter: 6h volume > 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(R4_6h[i]) or np.isnan(S4_6h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit: price returns below R4 or volume drops
            if high[i] < R4_6h[i] or not vol_ok:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price returns above S4 or volume drops
            if low[i] > S4_6h[i] or not vol_ok:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout beyond weekly R4/S4 with volume confirmation
            if vol_ok:
                if high[i] >= R4_6h[i] and close[i] > R4_6h[i]:  # Break above R4
                    position = 1
                    signals[i] = 0.25
                elif low[i] <= S4_6h[i] and close[i] < S4_6h[i]:  # Break below S4
                    position = -1
                    signals[i] = -0.25
    
    return signals