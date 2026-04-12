#!/usr/bin/env python3
"""
6h_1w_1d_Donchian_Breakout_Pivot_Direction_v1
Hypothesis: On 6h timeframe, enter long when price breaks above weekly Donchian upper band (20) and price is above daily pivot, enter short when price breaks below weekly Donchian lower band (20) and price is below daily pivot. Uses weekly price structure for direction and daily pivot for mean reversion filter. Designed for low trade frequency (<30/year) and robustness in both bull and bear markets by requiring confluence of weekly breakout and daily bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_Donchian_Breakout_Pivot_Direction_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === WEEKLY DONCHIAN CHANNELS ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 20-period Donchian channels
    donchian_high = np.full_like(high_1w, np.nan)
    donchian_low = np.full_like(low_1w, np.nan)
    
    for i in range(20, len(high_1w)):
        donchian_high[i] = np.max(high_1w[i-20:i])
        donchian_low[i] = np.min(low_1w[i-20:i])
    
    # Align to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # === DAILY PIVOT POINT ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot point
    pivot = (high_1d + low_1d + close_1d) / 3
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(pivot_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with daily pivot filter
        long_breakout = (close[i] > donchian_high_aligned[i]) and (close[i] > pivot_aligned[i])
        short_breakout = (close[i] < donchian_low_aligned[i]) and (close[i] < pivot_aligned[i])
        
        # Exit conditions: opposite Donchian breakout or price crosses pivot
        exit_long = (close[i] < donchian_low_aligned[i]) or (close[i] < pivot_aligned[i])
        exit_short = (close[i] > donchian_high_aligned[i]) or (close[i] > pivot_aligned[i])
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals