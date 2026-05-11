#!/usr/bin/env python3
name = "6h_Donchian20_WeeklyPivotDirection_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

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
    
    # Donchian channel on 6h (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly pivot from 1w data for direction filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Volume filter: 50-period EMA for spike detection
    vol_ema50 = pd.Series(volume).ewm(span=50, min_periods=50, adjust=False).mean().values
    volume_ok = volume > vol_ema50 * 2.0  # Volume spike filter
    
    # Fixed position size to minimize churn
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        breakout_long = close[i] > high_max[i]
        breakout_short = close[i] < low_min[i]
        price_above_pivot = close[i] > pivot_1w_aligned[i]
        price_below_pivot = close[i] < pivot_1w_aligned[i]
        
        if position == 0:
            # Long: Breakout above Donchian high + above weekly pivot + volume spike
            if breakout_long and price_above_pivot and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Breakout below Donchian low + below weekly pivot + volume spike
            elif breakout_short and price_below_pivot and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: Price closes below Donchian low OR below weekly pivot
                if close[i] < low_min[i] or close[i] < pivot_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price closes above Donchian high OR above weekly pivot
                if close[i] > high_max[i] or close[i] > pivot_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals