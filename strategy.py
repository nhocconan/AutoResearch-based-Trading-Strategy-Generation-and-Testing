#!/usr/bin/env python3
"""
6h Donchian breakout + weekly pivot direction + volume confirmation.
Hypothesis: Donchian breakouts capture trend continuation with high win rate.
Weekly pivot filters direction (only long above weekly pivot, short below).
Volume confirmation filters false breakouts. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14311_6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for weekly pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points from prior week
    # Need 5 days to form a week
    week_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(5)
    week_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(5)
    week_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().shift(5)
    
    # Weekly pivot formula: P = (H + L + C) / 3
    weekly_pivot = (week_high + week_low + week_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 5 for weekly pivot)
    start = max(20, 5) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: return to weekly pivot (mean reversion signal)
        if position == 1:  # long position
            if close[i] <= weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with weekly pivot filter and volume confirmation
            # Long when price breaks above Donchian high AND above weekly pivot with volume
            # Short when price breaks below Donchian low AND below weekly pivot with volume
            long_setup = (close[i] > donchian_high[i-1]) and (close[i] > weekly_pivot_aligned[i]) and vol_confirm[i]
            short_setup = (close[i] < donchian_low[i-1]) and (close[i] < weekly_pivot_aligned[i]) and vol_confirm[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals