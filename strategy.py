#!/usr/bin/env python3
"""
6h_Donchian_WeeklyPivot_Breakout_Trend
Hypothesis: 6h Donchian(20) breakouts in direction of weekly pivot bias (from 1d data)
with volume confirmation capture institutional flow. Weekly pivot provides structural
bias that works in bull/bear markets. Target: 15-30 trades/year on 6h to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot from prior week (using daily data)
    # Group into weeks: Monday to Sunday, use prior week's data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly high, low, close (using prior week)
    # We'll approximate: weekly high = max of last 5 daily highs, etc.
    # For simplicity, use 5-day period as proxy for weekly
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(5).values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(5).values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().shift(5).values
    
    # Weekly pivot point
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    # Weekly support/resistance levels (simplified)
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivot levels to 6h timeframe
    weekly_r1_1d_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_1d_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    weekly_pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # 6h Donchian channel (20 periods)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Donchian and volume MA
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(weekly_r1_1d_aligned[i]) or np.isnan(weekly_s1_1d_aligned[i]) or
            np.isnan(weekly_pivot_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        dh = donchian_high[i]
        dl = donchian_low[i]
        weekly_r1_val = weekly_r1_1d_aligned[i]
        weekly_s1_val = weekly_s1_1d_aligned[i]
        weekly_pivot_val = weekly_pivot_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with weekly bullish bias and volume spike
            if high[i] > dh and weekly_pivot_val > weekly_s1_val and vol_spike_val:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low with weekly bearish bias and volume spike
            elif low[i] < dl and weekly_pivot_val < weekly_r1_val and vol_spike_val:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls below Donchian low or weekly bias turns bearish
            if low[i] < dl or weekly_pivot_val < weekly_s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price rises above Donchian high or weekly bias turns bullish
            if high[i] > dh or weekly_pivot_val > weekly_r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian_WeeklyPivot_Breakout_Trend"
timeframe = "6h"
leverage = 1.0