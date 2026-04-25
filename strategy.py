#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 12h Weekly Pivot Direction + Volume Confirmation
Hypothesis: On 6h timeframe, price breaking Donchian(20) channels captures momentum.
Weekly pivot direction from 12h data filters for higher timeframe bias: only long when
weekly trend is up (price above weekly pivot), short when down.
Volume confirmation avoids false breakouts. Designed for low trade frequency (12-37/year)
to minimize fee drag. Works in bull/bear via pivot direction filter.
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
    
    # Get 12h data for weekly pivot (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # need ~1 week + buffer
        return np.zeros(n)
    
    # Calculate weekly pivot points on 12h data (using prior week's H/L/C)
    # Week = 7 days = 14 periods of 12h data
    df_12h = df_12h.copy()
    # Rolling window of 14 periods (2 weeks) to get prior week's H/L/C
    df_12h['week_high'] = df_12h['high'].rolling(window=14, min_periods=14).max().shift(14)
    df_12h['week_low'] = df_12h['low'].rolling(window=14, min_periods=14).min().shift(14)
    df_12h['week_close'] = df_12h['close'].rolling(window=14, min_periods=14).last().shift(14)
    
    # Weekly pivot = (H + L + C) / 3
    df_12h['weekly_pivot'] = (df_12h['week_high'] + df_12h['week_low'] + df_12h['week_close']) / 3
    # Weekly R1/S1 for context (optional)
    df_12h['weekly_range'] = df_12h['week_high'] - df_12h['week_low']
    df_12h['weekly_r1'] = df_12h['weekly_pivot'] + df_12h['weekly_range'] * 1.1 / 12
    df_12h['weekly_s1'] = df_12h['weekly_pivot'] - df_12h['weekly_range'] * 1.1 / 12
    
    weekly_pivot = df_12h['weekly_pivot'].values
    weekly_r1 = df_12h['weekly_r1'].values
    weekly_s1 = df_12h['weekly_s1'].values
    
    # Align weekly pivot to 6h timeframe (prior week's pivot known after week close)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_12h, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_12h, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_12h, weekly_s1)
    
    # 6h Donchian(20) channels
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # 6h volume MA(20) for confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, volume MA, and aligned data
    start_idx = max(lookback, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donchian_high = highest_high[i]
        donchian_low = lowest_low[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        weekly_r1_val = weekly_r1_aligned[i]
        weekly_s1_val = weekly_s1_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Volume confirmation: current 6h volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma_val
        
        if position == 0:
            # Look for entry signals
            # Long: Break above Donchian high AND price > weekly pivot (uptrend bias) AND volume confirmation
            long_entry = (curr_high > donchian_high and 
                         curr_close > weekly_pivot_val and volume_confirm)
            # Short: Break below Donchian low AND price < weekly pivot (downtrend bias) AND volume confirmation
            short_entry = (curr_low < donchian_low and 
                          curr_close < weekly_pivot_val and volume_confirm)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Price crosses below Donchian low OR weekly pivot breaks down
            if (curr_close < donchian_low or curr_close < weekly_pivot_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Price crosses above Donchian high OR weekly pivot breaks up
            if (curr_close > donchian_high or curr_close > weekly_pivot_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Direction_VolumeConfirm"
timeframe = "6h"
leverage = 1.0