#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Direction_V1
Hypothesis: 6-hour Donchian(20) breakouts in direction of weekly pivot (price > weekly pivot = long bias, < = short bias) with volume confirmation (1.5x average). Weekly pivot acts as dynamic support/resistance. Works in bull/bear by only taking breakouts aligned with weekly bias. Targets 15-35 trades/year by requiring weekly alignment + volume filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian channels: upper = max(high, period), lower = min(low, period)"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot: (high + low + close) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate 6h Donchian channels
    high = prices['high'].values
    low = prices['low'].values
    upper, lower = calculate_donchian_channels(high, low, 20)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if weekly pivot not ready
        if np.isnan(weekly_pivot_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper + price > weekly pivot + volume confirmation
            if price > upper[i] and price > weekly_pivot_aligned[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + price < weekly pivot + volume confirmation
            elif price < lower[i] and price < weekly_pivot_aligned[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian lower or price < weekly pivot
            if price < lower[i] or price < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian upper or price > weekly pivot
            if price > upper[i] or price > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Direction_V1"
timeframe = "6h"
leverage = 1.0