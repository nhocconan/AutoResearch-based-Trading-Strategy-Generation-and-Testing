#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeConfirmation
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation (>1.8x 20-period MA).
Long when price breaks above Donchian(20) high in weekly uptrend with volume spike.
Short when price breaks below Donchian(20) low in weekly downtrend with volume spike.
Weekly pivot direction uses the relationship between weekly close and weekly pivot point (PP = (H+L+C)/3).
This structure provides clean trend following with institutional-level reference points.
Designed for 12-37 trades/year on 6h timeframe. Works in both bull and bear markets by following weekly trend.
"""

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
    
    # Get weekly data for pivot direction filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot point and direction
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point: PP = (H + L + C) / 3
    weekly_pp = (high_1w + low_1w + close_1w) / 3.0
    # Weekly trend: close above PP = uptrend, below PP = downtrend
    weekly_uptrend = close_1w > weekly_pp
    weekly_downtrend = close_1w < weekly_pp
    
    # Align weekly trend to 6h timeframe
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    # Donchian(20) channels - calculated on 6h data
    # We need at least 20 periods for Donchian calculation
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.8x 20-period MA (balanced threshold)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian and volume MA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or 
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with weekly uptrend and volume spike
            if (close[i] > high_ma[i] and 
                weekly_uptrend_aligned[i] > 0.5 and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with weekly downtrend and volume spike
            elif (close[i] < low_ma[i] and 
                  weekly_downtrend_aligned[i] > 0.5 and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below Donchian low OR weekly trend changes to downtrend
            if (close[i] < low_ma[i] or weekly_downtrend_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above Donchian high OR weekly trend changes to uptrend
            if (close[i] > high_ma[i] or weekly_uptrend_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0