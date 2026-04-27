#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDir_VolumeConfirm
Hypothesis: 6h strategy using Donchian(20) breakouts with weekly pivot direction filter (from 1w HTF) and volume confirmation. Weekly pivot provides robust trend filter that works in both bull and bear markets by identifying institutional supply/demand zones. Volume confirmation ensures breakout has participation. Designed for low trade frequency (12-30/year) with discrete position sizing to minimize fee drag. Uses 6h primary timeframe with 1d/1w HTF for multi-timeframe alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channel calculation (20-period)
    df_1d = get_htf_data(prices, '1d')
    # Calculate Donchian(20) on 1d data: highest high and lowest low over 20 days
    highest_high = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    # Align to 6h timeframe (wait for completed 1d bar)
    donchian_high = align_htf_to_ltf(prices, df_1d, highest_high)
    donchian_low = align_htf_to_ltf(prices, df_1d, lowest_low)
    
    # Get 1w data for weekly pivot direction (more robust than daily)
    df_1w = get_htf_data(prices, '1w')
    # Weekly pivot: (weekly_high + weekly_low + weekly_close) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Align weekly pivot to 6h timeframe (wait for completed 1w bar)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Fixed position size to minimize churn
    
    # Warmup: need Donchian(20) (20), weekly pivot, volume avg (20)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with weekly pivot direction filter and volume confirmation
            long_condition = (close_val > donchian_high_val and 
                            close_val > weekly_pivot_val and  # Above weekly pivot = bullish bias
                            vol_conf)
            short_condition = (close_val < donchian_low_val and 
                             close_val < weekly_pivot_val and  # Below weekly pivot = bearish bias
                             vol_conf)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price closes below weekly pivot (trend change) or Donchian low (mean reversion)
            if close_val < weekly_pivot_val or close_val < donchian_low_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above weekly pivot (trend change) or Donchian high (mean reversion)
            if close_val > weekly_pivot_val or close_val > donchian_high_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDir_VolumeConfirm"
timeframe = "6h"
leverage = 1.0