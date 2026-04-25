#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation
Hypothesis: Weekly pivot levels (from prior week) establish major support/resistance.
Donchian(20) breakout in direction of weekly pivot bias (above weekly pivot = bullish bias,
below = bearish) with volume confirmation captures institutional breakouts.
Works in bull/bear: weekly pivot adapts to regime, volume filters false breakouts.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 1w data for weekly pivot (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot from prior week OHLC
    # Weekly Pivot = (PriorWeek High + PriorWeek Low + PriorWeek Close) / 3
    weekly_pivot = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate Donchian(20) on 6h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period volume MA for 6h volume confirmation
    vol_ma_20_6h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_6h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian and volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma_20_6h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        weekly_pivot_level = weekly_pivot_aligned[i]
        upper_donchian = highest_20[i]
        lower_donchian = lowest_20[i]
        vol_ma_6h = vol_ma_20_6h[i]
        
        # Volume confirmation: current 6h volume > 1.8 * 20-period average
        volume_confirm = curr_volume > 1.8 * vol_ma_20_6h
        
        if position == 0:
            # Look for entry signals
            # Long: price > weekly pivot (bullish bias) AND breaks above Donchian high AND volume confirmation
            long_entry = (curr_close > weekly_pivot_level and 
                         curr_high > upper_donchian and 
                         volume_confirm)
            # Short: price < weekly pivot (bearish bias) AND breaks below Donchian low AND volume confirmation
            short_entry = (curr_close < weekly_pivot_level and 
                          curr_low < lower_donchian and 
                          volume_confirm)
            
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
            # Exit: price falls below weekly pivot OR breaks below Donchian low (failed breakout)
            if curr_close < weekly_pivot_level or curr_low < lower_donchian:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above weekly pivot OR breaks above Donchian high (failed breakdown)
            if curr_close > weekly_pivot_level or curr_high > upper_donchian:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Direction_VolumeConfirm"
timeframe = "6h"
leverage = 1.0