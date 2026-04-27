#!/usr/bin/env python3
"""
#100799 - 6h_Donchian20_WeeklyPivot_Direction_Volume
Hypothesis: Donchian(20) breakout with weekly pivot direction filter and volume confirmation on 6h timeframe.
Rationale: Combines price breakout with weekly pivot bias to avoid counter-trend trades, works in both bull and bear markets.
Weekly pivot provides structural bias, Donchian captures breakouts, volume confirms conviction.
Targets 15-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for weekly pivot calculation (using daily to build weekly pivot)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot from prior week's daily data
    # Build weekly OHLC from daily data (simplified: use last 5 trading days)
    weekly_high = np.zeros(len(df_1d))
    weekly_low = np.zeros(len(df_1d))
    weekly_close = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        if i < 4:
            weekly_high[i] = np.nan
            weekly_low[i] = np.nan
            weekly_close[i] = np.nan
        else:
            # Simple weekly aggregation: last 5 days
            weekly_high[i] = np.max(df_1d[i-4:i+1])
            weekly_low[i] = np.min(df_1d[i-4:i+1])
            weekly_close[i] = df_1d['close'].iloc[i] if hasattr(df_1d, 'iloc') else df_1d[i]
    
    # Weekly pivot points
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivot to 6h timeframe (prior week's levels)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # Donchian channel (20-period) on 6h data
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume filter: volume > 1.8x 24-period average (4 days worth)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(30, lookback-1)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above Donchian high, above weekly pivot, volume spike
        if (close[i] > highest_high[i] and 
            close[i] > weekly_pivot_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below Donchian low, below weekly pivot, volume spike
        elif (close[i] < lowest_low[i] and 
              close[i] < weekly_pivot_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to weekly pivot (mean reversion to weekly bias)
        elif position == 1 and close[i] < weekly_pivot_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > weekly_pivot_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0