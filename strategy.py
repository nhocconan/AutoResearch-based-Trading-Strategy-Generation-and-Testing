#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
- Weekly pivot (from prior week) defines major trend: price above weekly pivot = bullish bias (longs only), below = bearish bias (shorts only).
- Donchian(20) breakout on 6h captures momentum in direction of weekly trend.
- Volume confirmation (>1.8x 20-bar average) ensures institutional participation.
- Position size 0.25 balances profit and drawdown control.
- Target trades: 60-120 total over 4 years (15-30/year) to minimize fee drag.
- Weekly pivot provides structural bias that works in both bull and bear markets by aligning with higher timeframe trend.
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
    
    # Get weekly data ONCE before loop for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly pivot from prior completed week
    prev_weekly_high = pd.Series(df_1w['high']).shift(1).values
    prev_weekly_low = pd.Series(df_1w['low']).shift(1).values
    prev_weekly_close = pd.Series(df_1w['close']).shift(1).values
    
    # Weekly pivot point (standard calculation)
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Donchian(20) channels on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20) + 1  # Need enough for Donchian and volume
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Only trade if volume confirms
            if volume_confirm:
                # Long breakout: price above Donchian high AND above weekly pivot (bullish bias)
                if close[i] > donchian_high[i] and close[i] > weekly_pivot_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakout: price below Donchian low AND below weekly pivot (bearish bias)
                elif close[i] < donchian_low[i] and close[i] < weekly_pivot_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low OR crosses below weekly pivot
            if close[i] < donchian_low[i] or close[i] < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high OR crosses above weekly pivot
            if close[i] > donchian_high[i] or close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0