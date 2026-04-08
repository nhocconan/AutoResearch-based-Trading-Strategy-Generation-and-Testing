#!/usr/bin/env python3
"""
6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation v1
Hypothesis: Donchian(20) breakouts filtered by weekly pivot direction (from 1d data) and volume confirmation capture sustained trends while avoiding whipsaws. The weekly pivot provides long-term structural bias, and volume validates breakout strength. This strategy targets 15-40 trades/year, balancing responsiveness with low turnover, and works in both bull and bear regimes by adapting to structural trends from higher timeframes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for weekly pivot calculation (using daily data to compute weekly pivot)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from daily data
    # Weekly high = max of last 5 daily highs
    weekly_high = pd.Series(df_1d['high'].values).rolling(window=5, min_periods=5).max().values
    # Weekly low = min of last 5 daily lows
    weekly_low = pd.Series(df_1d['low'].values).rolling(window=5, min_periods=5).min().values
    # Weekly close = last daily close
    weekly_close = df_1d['close'].values
    # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # 6h Donchian Channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or weekly pivot shifts bearish
            if close[i] <= donchian_low[i] or close[i] < weekly_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or weekly pivot shifts bullish
            if close[i] >= donchian_high[i] or close[i] > weekly_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout with bullish weekly pivot bias and volume
            if (close[i] >= donchian_high[i] and 
                close[i] > weekly_pivot_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown with bearish weekly pivot bias and volume
            elif (close[i] <= donchian_low[i] and 
                  close[i] < weekly_pivot_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals