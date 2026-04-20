#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian Breakout with Weekly Pivot Direction Filter
# - Use 60-period Donchian channel on 6h for breakout signals
# - Filter trades by weekly pivot direction (long above weekly pivot, short below)
# - Volume confirmation: require volume > 1.5x 20-period average
# - Weekly pivot provides institutional reference point, filters counter-trend noise
# - Designed for 6h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation
    df_w = get_htf_data(prices, '1w')
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    pivot_w = (high_w + low_w + close_w) / 3
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    
    # Align weekly pivot to 6h
    pivot_w_6h = align_htf_to_ltf(prices, df_w, pivot_w)
    r1_w_6h = align_htf_to_ltf(prices, df_w, r1_w)
    s1_w_6h = align_htf_to_ltf(prices, df_w, s1_w)
    r2_w_6h = align_htf_to_ltf(prices, df_w, r2_w)
    s2_w_6h = align_htf_to_ltf(prices, df_w, s2_w)
    
    # Calculate 60-period Donchian channel on 6h
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    donchian_high = pd.Series(high_6h).rolling(window=60, min_periods=60).max().values
    donchian_low = pd.Series(low_6h).rolling(window=60, min_periods=60).min().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume_6h > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if NaN in indicators
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(pivot_w_6h[i]) or np.isnan(r1_w_6h[i]) or np.isnan(s1_w_6h[i]) or \
           np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        breakout_up = close_6h[i] > donchian_high[i-1]  # Break above prior high
        breakout_down = close_6h[i] < donchian_low[i-1]  # Break below prior low
        
        # Weekly pivot direction filter
        above_pivot = close_6h[i] > pivot_w_6h[i]
        below_pivot = close_6h[i] < pivot_w_6h[i]
        
        if position == 0:
            # Long entry: upward breakout + above weekly pivot + volume
            if breakout_up and above_pivot and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: downward breakout + below weekly pivot + volume
            elif breakout_down and below_pivot and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or falls below weekly pivot
            if close_6h[i] < donchian_low[i] or close_6h[i] < pivot_w_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or rises above weekly pivot
            if close_6h[i] > donchian_high[i] or close_6h[i] > pivot_w_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_WeeklyPivot_DirectionFilter"
timeframe = "6h"
leverage = 1.0