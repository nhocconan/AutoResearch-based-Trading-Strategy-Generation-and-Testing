#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
# Long when price breaks above Donchian upper band (20-period) AND weekly pivot above weekly close AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower band (20-period) AND weekly pivot below weekly close AND volume > 1.5x 20-period average.
# Exit when price crosses back inside the Donchian channel (between upper and lower bands).
# Donchian provides trend-following structure, weekly pivot filters higher timeframe bias, volume confirms institutional participation.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Donchian_20_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot calculation
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot from previous week's OHLC
    prev_weekly_close = df_w['close'].shift(1).values
    prev_weekly_high = df_w['high'].shift(1).values
    prev_weekly_low = df_w['low'].shift(1).values
    pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3
    
    # Align weekly pivot to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_w, pivot)
    
    # Donchian channels (20-period) on 6h data
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)  # Sufficient warmup for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper, weekly pivot above weekly close, volume filter
            long_cond = (close[i] > donchian_upper[i]) and (pivot_aligned[i] > prev_weekly_close[i-1] if i > 0 else False) and volume_filter[i]
            # Short conditions: price breaks below Donchian lower, weekly pivot below weekly close, volume filter
            short_cond = (close[i] < donchian_lower[i]) and (pivot_aligned[i] < prev_weekly_close[i-1] if i > 0 else False) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian lower band
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Donchian upper band
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals