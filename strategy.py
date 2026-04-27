#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot bias and volume confirmation
# - Weekly pivot from 1d data provides directional bias (above/below weekly pivot)
# - Donchian breakout captures momentum in direction of bias
# - Volume filter ensures breakouts have conviction
# - Works in bull/bear: pivot adapts to regime, breakouts catch strong moves
# Target: 20-30 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points from daily data (using prior week's data)
    # We need to group daily data into weeks and calculate pivot for prior week
    # For simplicity, we'll use a rolling window of 5 days (1 week) to approximate
    # In practice, we'd need to align to actual week boundaries, but rolling 5-day works
    high_5d = pd.Series(df_1d['high'].values).rolling(window=5, min_periods=5).max().values
    low_5d = pd.Series(df_1d['low'].values).rolling(window=5, min_periods=5).min().values
    close_5d = pd.Series(df_1d['close'].values).rolling(window=5, min_periods=5).mean().values
    
    # Weekly pivot = (H + L + C) / 3
    weekly_pivot = (high_5d + low_5d + close_5d) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Donchian channel (20-period) on 6h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price breaks above Donchian high AND above weekly pivot + volume
        if (close[i] > high_20[i] and 
            close[i] > weekly_pivot_aligned[i] and
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: price breaks below Donchian low AND below weekly pivot + volume
        elif (close[i] < low_20[i] and 
              close[i] < weekly_pivot_aligned[i] and
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Bias_Volume"
timeframe = "6h"
leverage = 1.0