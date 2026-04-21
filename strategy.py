#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotTrend_VolumeFilter
Hypothesis: 6h Donchian(20) breakouts aligned with weekly pivot trend (price above/below weekly pivot) and volume confirmation.
Weekly pivot acts as institutional reference point - price above weekly pivot indicates bullish bias, below indicates bearish bias.
Volume spike (>1.5x 20-period average) confirms breakout strength.
Designed for low trade frequency (12-30 trades/year) to minimize fee drag.
Works in bull/bear via weekly pivot alignment and volume confirmation filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for weekly pivot calculation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Weekly Pivot Point calculation from daily data ===
    # Weekly pivot = (Weekly High + Weekly Low + Weekly Close) / 3
    # We'll approximate using rolling window on daily data (5 trading days ≈ 1 week)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly high, low, close using 5-day rolling window
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot point
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # === 6h Donchian Channel (20-period) ===
    # We need to calculate Donchian on 6h data directly from prices
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma  # Pre-compute boolean array
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) 
            or np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        
        if position == 0:
            # Volume spike condition
            if not vol_spike[i]:
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                continue
            
            # Long conditions: price breaks above 6h Donchian high AND price > weekly pivot
            long_breakout = price > donchian_high[i]
            long_trend = price > weekly_pivot_aligned[i]
            
            # Short conditions: price breaks below 6h Donchian low AND price < weekly pivot
            short_breakout = price < donchian_low[i]
            short_trend = price < weekly_pivot_aligned[i]
            
            # Entry logic
            if long_breakout and long_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions: price closes below weekly pivot OR Donchian low broken
            if price < weekly_pivot_aligned[i] or price < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: price closes above weekly pivot OR Donchian high broken
            if price > weekly_pivot_aligned[i] or price > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0