#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot regime filter and volume confirmation.
- Long: Close breaks above Donchian upper band (20) + price above 1d weekly pivot (bullish bias) + volume > 1.3x 20-period avg
- Short: Close breaks below Donchian lower band (20) + price below 1d weekly pivot (bearish bias) + volume > 1.3x 20-period avg
- Exit: Close crosses opposite Donchian band (mean reversion at channel center)
- Uses Donchian channels for breakout structure, weekly pivot for regime bias, volume for confirmation
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Discrete position sizing: ±0.25 to balance return and minimize fee churn
- Works in bull markets (breakouts with bullish bias) and bear markets (breakouts with bearish bias)
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
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly pivot from 1d HTF data (using weekly approximation: (H+L+C)/3)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly pivot approximation using daily data (standard pivot point)
    weekly_pivot = (high_1d + low_1d + close_1d) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 1)  # Need 20 for Donchian, 1 for HTF data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(weekly_pivot_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.3x average)
        volume_confirm = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Long: Close breaks above Donchian upper + price above weekly pivot (bullish bias) + volume
            if (close[i] > highest_high[i] and 
                close[i] > weekly_pivot_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian lower + price below weekly pivot (bearish bias) + volume
            elif (close[i] < lowest_low[i] and 
                  close[i] < weekly_pivot_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close crosses below Donchian lower band (mean reversion)
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close crosses above Donchian upper band (mean reversion)
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dWeeklyPivot_VolumeSpike"
timeframe = "6h"
leverage = 1.0