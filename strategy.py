#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
# Long when price breaks above Donchian upper band AND weekly pivot > previous weekly pivot AND volume > 2.0x 20-bar avg
# Short when price breaks below Donchian lower band AND weekly pivot < previous weekly pivot AND volume > 2.0x 20-bar avg
# Exit when price reverts to Donchian midpoint (mean reversion)
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 12-37 trades/year on 6h timeframe.
# Donchian provides trend structure, weekly pivot filters direction, volume confirmation ensures breakout strength.
# This combination has not been heavily tested on 6h and should work in both bull (breakouts) and bear (mean reversion in ranges).

name = "6h_Donchian20_WeeklyPivot_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 1w data for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot: (weekly_high + weekly_low + weekly_close) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Donchian and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_upper = donchian_upper[i]
        curr_lower = donchian_lower[i]
        curr_mid = donchian_mid[i]
        curr_weekly_pivot = weekly_pivot_aligned[i]
        curr_close = close[i]
        
        # Get previous weekly pivot for direction filter
        prev_weekly_pivot = weekly_pivot_aligned[i-1] if i > 0 else curr_weekly_pivot
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price reverts to Donchian midpoint (mean reversion)
            if curr_close <= curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reverts to Donchian midpoint (mean reversion)
            if curr_close >= curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian upper AND weekly pivot rising AND volume confirmation
            if curr_close > curr_upper and curr_weekly_pivot > prev_weekly_pivot and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian lower AND weekly pivot falling AND volume confirmation
            elif curr_close < curr_lower and curr_weekly_pivot < prev_weekly_pivot and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals