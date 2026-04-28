#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivotDirection_VolumeConfirmation
Hypothesis: Combine 6-hour Donchian breakout with weekly pivot direction as trend filter and volume confirmation.
Only take long positions when price breaks above Donchian upper AND weekly pivot shows bullish bias (price above weekly pivot).
Only take short positions when price breaks below Donchian lower AND weekly pivot shows bearish bias (price below weekly pivot).
Weekly pivot provides longer-term trend context to avoid counter-trend trades.
Volume confirmation ensures breakouts have institutional participation.
Designed to work in both bull and bear markets by following the weekly trend.
Target: 50-150 total trades over 4 years = 12-37/year.
"""

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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's data)
    # Pivot = (H + L + C) / 3
    # Support 1 = (2*Pivot) - H
    # Resistance 1 = (2*Pivot) - L
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    prev_weekly_close = df_1w['close'].shift(1).values
    
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    weekly_r1 = (2 * weekly_pivot) - prev_weekly_high
    weekly_s1 = (2 * weekly_pivot) - prev_weekly_low
    
    # Get daily data for volume context (optional)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels (20-period) on 6h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Align weekly pivot data to 6h
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Trend filter based on weekly pivot
    # Bullish bias: price above weekly pivot
    # Bearish bias: price below weekly pivot
    weekly_bullish = close > weekly_pivot_aligned
    weekly_bearish = close < weekly_pivot_aligned
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, lookback)  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        # Long: price breaks above Donchian upper + weekly bullish + volume surge
        long_entry = (close[i] > highest_high[i] and 
                     weekly_bullish[i] and 
                     volume_surge[i])
        
        # Short: price breaks below Donchian lower + weekly bearish + volume surge
        short_entry = (close[i] < lowest_low[i] and 
                      weekly_bearish[i] and 
                      volume_surge[i])
        
        # Exit when price crosses weekly pivot in opposite direction
        long_exit = weekly_bearish[i]  # Price crossed below weekly pivot
        short_exit = weekly_bullish[i]  # Price crossed above weekly pivot
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Donchian20_WeeklyPivotDirection_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0