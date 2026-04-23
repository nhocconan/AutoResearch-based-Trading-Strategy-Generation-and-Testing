#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
- Primary timeframe: 6h, HTF: 1w for pivot direction, 1d for volume confirmation
- Long: Close breaks above Donchian upper(20) + price > weekly pivot (bullish bias) + volume > 2.0x 20-period avg
- Short: Close breaks below Donchian lower(20) + price < weekly pivot (bearish bias) + volume > 2.0x 20-period avg
- Exit: Close reverts to Donchian midpoint (mean of upper/lower)
- Uses weekly pivot for structural bias to avoid counter-trend trades in strong weekly trends
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull markets (breakouts with bullish weekly bias) and bear markets (breakdowns with bearish weekly bias)
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
    
    # Volume confirmation: > 2.0x 20-period average (volume spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate weekly pivot points (from 1w data)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point: (H+L+C)/3
    weekly_pp = (high_1w + low_1w + close_1w) / 3.0
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 1)  # Need 20 for Donchian/volume MA, 1 for weekly pivot (aligned from 1w)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(weekly_pp_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Close breaks above Donchian upper + price > weekly PP (bullish bias) + volume spike
            if (close[i] > donchian_upper[i] and 
                close[i] > weekly_pp_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian lower + price < weekly PP (bearish bias) + volume spike
            elif (close[i] < donchian_lower[i] and 
                  close[i] < weekly_pp_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close reverts to Donchian midpoint
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close reverts to Donchian midpoint
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_VolumeSpike"
timeframe = "6h"
leverage = 1.0