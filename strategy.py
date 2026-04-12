#!/usr/bin/env python3
"""
6h_1d_wkly_pivot_volume_fade_v1
Hypothesis: Fade at weekly pivot extremes (R4/S4) with volume confirmation on 6h timeframe.
Weekly pivot levels act as strong support/resistance. Fading these extremes with volume
confirmation works in both bull and bear markets as price tends to revert from overextended levels.
Uses 1d data for weekly pivot calculation to avoid look-ahead. Target: 20-40 trades/year.
"""

name = "6h_1d_wkly_pivot_volume_fade_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # Get daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points using previous week's data
    # Need to group daily data into weeks
    weeks_high = []
    weeks_low = []
    weeks_close = []
    
    # Simple approach: use rolling window of 5 days (1 week) for pivot calculation
    # This avoids complex resampling and uses available 1d data
    if len(high_1d) >= 5:
        # Rolling window of 5 days for weekly high/low/close
        week_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
        week_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
        week_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
        
        # Calculate pivot points for previous week (shift by 1 to avoid look-ahead)
        prev_week_high = np.roll(week_high, 1)
        prev_week_low = np.roll(week_low, 1)
        prev_week_close = np.roll(week_close, 1)
        
        # Weekly pivot calculation
        pp = (prev_week_high + prev_week_low + prev_week_close) / 3.0
        r1 = 2 * pp - prev_week_low
        r2 = pp + (prev_week_high - prev_week_low)
        r3 = prev_week_high + 2 * (pp - prev_week_low)
        r4 = prev_week_high + 3 * (pp - prev_week_low)  # Extreme resistance
        
        s1 = 2 * pp - prev_week_high
        s2 = pp - (prev_week_high - prev_week_low)
        s3 = prev_week_low - 2 * (prev_week_high - pp)
        s4 = prev_week_low - 3 * (prev_week_high - pp)  # Extreme support
        
        # Align weekly pivot levels to 6h timeframe
        pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        # Not enough data for weekly calculation
        pp_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(pp_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price near S4 support with volume confirmation (fade extreme)
        if (close[i] <= s4_aligned[i] * 1.005 and  # Allow small buffer
            vol_confirm[i] and 
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price near R4 resistance with volume confirmation (fade extreme)
        elif (close[i] >= r4_aligned[i] * 0.995 and  # Allow small buffer
              vol_confirm[i] and 
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price returns to pivot point or opposite extreme
        elif position == 1 and close[i] >= pp_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] <= pp_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals