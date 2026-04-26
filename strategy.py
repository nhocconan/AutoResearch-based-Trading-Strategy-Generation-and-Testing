#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_Direction_v1
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter (1w) captures strong directional moves while avoiding counter-trend whipsaw. Uses volume confirmation (>1.5x average) and discrete sizing (0.25) to limit trades to 12-37/year. Weekly pivot acts as HTF trend filter: only long when price > weekly pivot, short when price < weekly pivot. Works in both bull/bear markets by aligning with dominant weekly trend.
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
    
    # Load weekly data ONCE before loop for pivot and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly pivot: (weekly high + weekly low + weekly close) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Donchian(20) on 6h timeframe
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of Donchian lookback (20), volume MA (20)
    start_idx = max(lookback, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        highest_high_val = highest_high[i]
        lowest_low_val = lowest_low[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Skip if any data not ready
        if (np.isnan(highest_high_val) or np.isnan(lowest_low_val) or 
            np.isnan(weekly_pivot_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Donchian breakout conditions
        breakout_up = close_val > highest_high_val
        breakout_down = close_val < lowest_low_val
        
        # Weekly pivot direction filter: price > pivot = bullish bias, price < pivot = bearish bias
        is_bullish_bias = close_val > weekly_pivot_val
        is_bearish_bias = close_val < weekly_pivot_val
        
        # Entry conditions: Donchian breakout in direction of weekly pivot + volume
        long_condition = breakout_up and is_bullish_bias and vol_conf
        short_condition = breakout_down and is_bearish_bias and vol_conf
        
        # Exit conditions: opposite Donchian breakout or weekly pivot flip
        long_exit = (position == 1 and (breakout_down or not is_bullish_bias))
        short_exit = (position == -1 and (breakout_up or not is_bearish_bias))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_v1"
timeframe = "6h"
leverage = 1.0