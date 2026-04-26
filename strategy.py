#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dTrend_Filter_v1
Hypothesis: 6h Donchian(20) breakout confirmed by 1d EMA50 trend direction. Uses discrete position sizing (0.25) and exits on opposite Donchian break. Designed for low trade frequency (12-37/year) to minimize fee drag while capturing medium-term trends in both bull and bear markets via trend filter.
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
    
    # Calculate Donchian channels (20-period) on 6h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for Donchian, 50 for 1d EMA
    start_idx = max(lookback, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for breakout in direction of 1d trend
            # Long: price breaks above 20-period high AND 1d EMA50 is rising
            long_entry = (close_val > highest_high[i]) and (ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1])
            # Short: price breaks below 20-period low AND 1d EMA50 is falling
            short_entry = (close_val < lowest_low[i]) and (ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1])
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on break below 20-period low (contrarian exit)
            if close_val < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on break above 20-period high (contrarian exit)
            if close_val > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_1dTrend_Filter_v1"
timeframe = "6h"
leverage = 1.0