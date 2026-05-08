#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation.
# Long when price breaks above 6h Donchian upper band AND weekly pivot > previous weekly pivot AND volume > 1.5x 20-period average.
# Short when price breaks below 6h Donchian lower band AND weekly pivot < previous weekly pivot AND volume > 1.5x 20-period average.
# Exit when price crosses back inside the 6h Donchian channel (between upper and lower bands).
# Weekly pivot provides institutional bias from higher timeframe. Donchian breakout captures momentum.
# Volume filter confirms institutional participation. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Donchian_20_1wPivot_Volume"
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
    
    # 1w data for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot point: (H + L + C) / 3 from previous week
    weekly_high = df_1w['high'].shift(1).values
    weekly_low = df_1w['low'].shift(1).values
    weekly_close = df_1w['close'].shift(1).values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Weekly pivot direction: rising if current > previous, falling if current < previous
    pivot_rising = np.zeros_like(weekly_pivot_aligned, dtype=bool)
    pivot_falling = np.zeros_like(weekly_pivot_aligned, dtype=bool)
    pivot_rising[1:] = weekly_pivot_aligned[1:] > weekly_pivot_aligned[:-1]
    pivot_falling[1:] = weekly_pivot_aligned[1:] < weekly_pivot_aligned[:-1]
    
    # 6h Donchian channel (20-period)
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        upper[i] = np.max(high[i - lookback + 1:i + 1])
        lower[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 2)  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(pivot_rising[i]) or np.isnan(pivot_falling[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper band, weekly pivot rising, volume filter
            long_cond = (close[i] > upper[i]) and pivot_rising[i] and volume_filter[i]
            # Short conditions: price breaks below lower band, weekly pivot falling, volume filter
            short_cond = (close[i] < lower[i]) and pivot_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below lower band (mean reversion)
            if close[i] < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above upper band (mean reversion)
            if close[i] > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals