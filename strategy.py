#!/usr/bin/env python3
"""
6h_Donchian_Breakout_WeeklyPivot_Direction_Volume
Hypothesis: Price breaking above/below 6h Donchian(20) channels, with weekly pivot direction (from weekly high/low/close) as trend filter and volume confirmation, captures institutional breakouts while avoiding false signals. Works in bull/bear by following higher timeframe trend. Low frequency via 6h timeframe and strict entry criteria.
Target: 50-150 total trades over 4 years (12-37/year).
"""
name = "6h_Donchian_Breakout_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot direction
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot point (based on previous week's high, low, close)
    prev_weekly_high = df_w['high'].shift(1).values
    prev_weekly_low = df_w['low'].shift(1).values
    prev_weekly_close = df_w['close'].shift(1).values
    
    # Weekly pivot point
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3
    
    # Align weekly pivot to 6h timeframe (values available after weekly bar closes)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_w, weekly_pivot)
    
    # Donchian channel (20 periods) on 6h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need Donchian and volume average
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + above weekly pivot + volume
            if close[i] > donchian_high[i] and close[i] > weekly_pivot_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + below weekly pivot + volume
            elif close[i] < donchian_low[i] and close[i] < weekly_pivot_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to opposite Donchian level (mean reversion)
            if position == 1:
                if close[i] <= donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] >= donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals