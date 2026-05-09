#!/usr/bin/env python3
# Hypothesis: 12h timeframe with weekly Donchian breakout and daily volume filter.
# Uses weekly Donchian channel (20-period) for structural breakouts and daily volume > 1.5x 20-period average for confirmation.
# Weekly Donchian provides robust breakout levels that work in both trending and ranging markets.
# Daily volume filter ensures breakouts are supported by participation, reducing false signals.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "12h_Donchian_Breakout_1wVolFilter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate weekly Donchian channel (20-period) from prior week
    # 20 weeks * 12h bars per week = 20 * 14 = 280 bars
    lookback = 280
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        highest_high[i] = np.max(high[i-lookback:i])
        lowest_low[i] = np.min(low[i-lookback:i])
    
    # Breakout conditions
    breakout_up = close > highest_high
    breakout_down = close < lowest_low
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period average volume on daily timeframe
    avg_volume_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Volume filter: current volume > 1.5x 20-period average volume
    volume_filter = volume > (1.5 * avg_volume_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 280  # Need enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above weekly Donchian high + volume filter
            if breakout_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below weekly Donchian low + volume filter
            elif breakout_down[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to weekly Donchian low or opposite breakout
            if close[i] <= lowest_low[i] or breakout_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to weekly Donchian high or opposite breakout
            if close[i] >= highest_high[i] or breakout_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals