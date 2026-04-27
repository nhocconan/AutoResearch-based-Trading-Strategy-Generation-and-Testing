#!/usr/bin/env python3
"""
#100947 - 6h_Donchian20_WeeklyPivot_Direction_VolumeConfirmation
Hypothesis: Use weekly pivot direction (from Monday's close) to bias breakouts on Donchian(20) channels at 6h timeframe. Enter long when price breaks above Donchian upper band AND weekly bias is bullish (price above weekly pivot). Enter short when price breaks below Donchian lower band AND weekly bias is bearish (price below weekly pivot). Volume confirmation required. Targets 15-35 trades/year to minimize fee drag. Weekly pivot provides structural bias that works in both bull (breakouts with trend) and bear (mean reversion to pivot) markets.
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
    
    # Get weekly data for pivot direction bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot from previous week (to avoid look-ahead)
    weekly_pivot = (high_1w + low_1w + close_1w) / 3
    
    # Align weekly pivot to 6h timeframe (previous week's pivot for current period)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above Donchian upper band, above weekly pivot (bullish bias), volume spike
        if (close[i] > highest_high[i] and 
            close[i] > weekly_pivot_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below Donchian lower band, below weekly pivot (bearish bias), volume spike
        elif (close[i] < lowest_low[i] and 
              close[i] < weekly_pivot_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to weekly pivot (mean reversion to pivot)
        elif position == 1 and close[i] < weekly_pivot_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > weekly_pivot_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Direction_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0