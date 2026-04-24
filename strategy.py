#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
- Long when price breaks above Donchian upper band (20-period high) AND weekly pivot > previous weekly pivot (bullish weekly bias) AND volume > 1.5 * median volume of last 20 bars
- Short when price breaks below Donchian lower band (20-period low) AND weekly pivot < previous weekly pivot (bearish weekly bias) AND volume > 1.5 * median volume of last 20 bars
- Exit on opposite Donchian breakout or when weekly pivot bias flips
- Uses 6h primary timeframe with 1w HTF to target 50-150 total trades over 4 years (12-37/year)
- Donchian channels provide clear breakout levels that work in both trending and ranging markets
- Weekly pivot adds higher timeframe structure to avoid counter-trend whipsaws
- Volume confirmation reduces fakeouts
- Designed for BTC/ETH with edge in capturing sustained moves aligned with weekly momentum
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
    
    # Calculate Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1w data ONCE before loop for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot: (weekly_high + weekly_low + weekly_close) / 3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Previous weekly pivot for bias detection
    weekly_pivot_prev = np.roll(weekly_pivot_aligned, 1)
    weekly_pivot_prev[0] = np.nan
    
    # Volume confirmation: volume > 1.5 * median volume of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (1.5 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 2) + 1  # Donchian needs 20, weekly pivot needs 2
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_pivot_prev[i]) or 
            np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper, weekly pivot rising (bullish bias), volume confirmation
            if close[i] > donchian_upper[i] and weekly_pivot_aligned[i] > weekly_pivot_prev[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower, weekly pivot falling (bearish bias), volume confirmation
            elif close[i] < donchian_lower[i] and weekly_pivot_aligned[i] < weekly_pivot_prev[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian lower OR weekly pivot falls (bearish bias flip)
            if close[i] < donchian_lower[i] or weekly_pivot_aligned[i] < weekly_pivot_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian upper OR weekly pivot rises (bullish bias flip)
            if close[i] > donchian_upper[i] or weekly_pivot_aligned[i] > weekly_pivot_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0