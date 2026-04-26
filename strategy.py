#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_Direction_v1
Hypothesis: 6h Donchian(20) breakouts in the direction of weekly pivot trend (price > weekly pivot = long bias, price < weekly pivot = short bias) with volume confirmation (>1.5x 20-bar average) capture strong trending moves. Weekly pivot provides structural support/resistance from higher timeframe, reducing false breakouts. Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year). Works in bull/bear by aligning with weekly pivot direction. Volume confirmation ensures momentum validity.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard: P = (H+L+C)/3)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of Donchian (20), volume MA (20)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        dh = donchian_high[i]
        dl = donchian_low[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Skip if any data not ready
        if (np.isnan(dh) or np.isnan(dl) or np.isnan(weekly_pivot_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Donchian breakout conditions
        long_breakout = close_val > dh
        short_breakout = close_val < dl
        
        # Weekly pivot trend filter: price > weekly pivot = long bias, price < weekly pivot = short bias
        is_long_bias = close_val > weekly_pivot_val
        is_short_bias = close_val < weekly_pivot_val
        
        # Entry conditions: Donchian breakout in direction of weekly pivot bias + volume confirmation
        long_entry = long_breakout and is_long_bias and vol_conf
        short_entry = short_breakout and is_short_bias and vol_conf
        
        # Update highest/lowest for trailing stop (using Donchian channels as dynamic stop)
        if position == 1:
            highest_since_long = max(highest_since_long, high_val)
        elif position == -1:
            lowest_since_short = min(lowest_since_short, low_val)
        elif position == 0:
            highest_since_long = 0.0
            lowest_since_short = 0.0
        
        # Exit conditions: Donchian channel reversal (opposite channel touch)
        long_exit = False
        short_exit = False
        if position == 1:
            # Long exit: price touches or crosses below Donchian low
            long_exit = low_val <= dl
        elif position == -1:
            # Short exit: price touches or crosses above Donchian high
            short_exit = high_val >= dh
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            highest_since_long = high_val
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            lowest_since_short = low_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
            highest_since_long = 0.0
        elif short_exit:
            signals[i] = 0.0
            position = 0
            lowest_since_short = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_v1"
timeframe = "6h"
leverage = 1.0