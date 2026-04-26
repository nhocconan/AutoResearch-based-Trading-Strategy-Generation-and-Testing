#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dTrend_Filter
Hypothesis: Use 6h timeframe with Donchian(20) breakout confirmed by 1d EMA50 trend direction. 
Enter long when price breaks above 20-period high AND 1d EMA50 is rising. 
Enter short when price breaks below 20-period low AND 1d EMA50 is falling.
Exit on opposite Donchian break (10-period) to reduce whipsaw.
Works in both bull and bear markets by following the 1d trend filter.
Target: 12-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate 1d EMA50 for trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels on 6h data
    # Upper band: 20-period high
    # Lower band: 20-period low
    # Exit bands: 10-period for faster exit
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for Donchian, 50 for 1d EMA
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or
            np.isnan(donchian_high_10[i]) or np.isnan(donchian_low_10[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # 25% position size to manage risk
        
        if position == 0:
            # Flat - look for breakout in direction of 1d EMA50 trend
            # Long: price breaks above 20-period high AND 1d EMA50 is rising
            long_breakout = close_val > donchian_high_20[i]
            ema_rising = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            
            # Short: price breaks below 20-period low AND 1d EMA50 is falling
            short_breakout = close_val < donchian_low_20[i]
            ema_falling = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]
            
            if long_breakout and ema_rising:
                signals[i] = size
                position = 1
            elif short_breakout and ema_falling:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price breaks below 10-period low (faster exit)
            if close_val < donchian_low_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above 10-period high (faster exit)
            if close_val > donchian_high_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0