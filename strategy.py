#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeConfirmation
Hypothesis: 6h Donchian(20) breakout aligned with weekly Camarilla pivot direction (R4/S4) and volume confirmation.
Enters long when price breaks above 6h Donchian(20) high with weekly bullish bias (close > weekly R4) and volume spike.
Enters short when price breaks below 6h Donchian(20) low with weekly bearish bias (close < weekly S4) and volume spike.
Exits when price reverses to opposite Donchian level or weekly bias flips.
Uses weekly HTF for structural bias to avoid counter-trend trades in both bull and bear markets.
Target: 12-37 trades/year on 6h (50-150 total over 4 years).
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
    
    # Get weekly data for Camarilla pivot calculation (structural bias)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (R4, S4) from previous week
    high_1w_prev = np.roll(df_1w['high'].values, 1)
    low_1w_prev = np.roll(df_1w['low'].values, 1)
    close_1w_prev = np.roll(df_1w['close'].values, 1)
    # First value will be invalid (rolled from last), set to nan
    high_1w_prev[0] = np.nan
    low_1w_prev[0] = np.nan
    close_1w_prev[0] = np.nan
    
    # Weekly Camarilla pivot calculation
    pivot_1w = (high_1w_prev + low_1w_prev + close_1w_prev) / 3.0
    range_1w = high_1w_prev - low_1w_prev
    r4_1w = pivot_1w + (range_1w * 1.0 / 2.0)  # R4 level
    s4_1w = pivot_1w - (range_1w * 1.0 / 2.0)  # S4 level
    
    # Align weekly Camarilla levels to 6h timeframe
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Weekly close for bias (close > R4 = bullish, close < S4 = bearish)
    close_1w_prev = df_1w['close'].values
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w_prev)
    
    # 6h Donchian(20) channels
    lookback = 20
    highest = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian, 20 for volume MA, 1 for weekly alignment)
    start_idx = max(lookback, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest[i]) or np.isnan(lowest[i]) or 
            np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or 
            np.isnan(close_1w_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with weekly bullish bias and volume spike
            if (close[i] > highest[i] and 
                close_1w_aligned[i] > r4_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with weekly bearish bias and volume spike
            elif (close[i] < lowest[i] and 
                  close_1w_aligned[i] < s4_1w_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below Donchian low OR weekly bias turns bearish
            if (close[i] < lowest[i] or close_1w_aligned[i] < s4_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above Donchian high OR weekly bias turns bullish
            if (close[i] > highest[i] or close_1w_aligned[i] > r4_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0