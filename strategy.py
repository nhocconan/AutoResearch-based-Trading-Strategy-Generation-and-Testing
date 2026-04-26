#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Direction_VolumeConfirmation
Hypothesis: Trade 6h Donchian(20) breakouts in the direction of the weekly pivot trend (price above/below weekly pivot),
confirmed by volume spikes. Weekly pivot provides structural bias for both bull and bear markets.
Donchian breakouts capture momentum, while weekly pivot filter avoids counter-trend trades.
Volume confirmation ensures breakout validity. Target: 50-150 trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot from previous weekly bar
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    prev_weekly_close = df_1w['close'].shift(1).values
    
    # Avoid NaN from shift
    prev_weekly_high = np.where(np.isnan(prev_weekly_high), df_1w['high'].values, prev_weekly_high)
    prev_weekly_low = np.where(np.isnan(prev_weekly_low), df_1w['low'].values, prev_weekly_low)
    prev_weekly_close = np.where(np.isnan(prev_weekly_close), df_1w['close'].values, prev_weekly_close)
    
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate Donchian(20) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Donchian(20), volume MA
    start_idx = max(20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        above_weekly_pivot = close_val > weekly_pivot_aligned[i]
        below_weekly_pivot = close_val < weekly_pivot_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND above weekly pivot AND volume spike
            long_signal = (close_val > donchian_high[i]) and above_weekly_pivot and vol_spike
            
            # Short: price breaks below Donchian low AND below weekly pivot AND volume spike
            short_signal = (close_val < donchian_low[i]) and below_weekly_pivot and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below Donchian low OR weekly pivot flips (close below pivot)
            if (close_val < donchian_low[i]) or (not above_weekly_pivot):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above Donchian high OR weekly pivot flips (close above pivot)
            if (close_val > donchian_high[i]) or (not below_weekly_pivot):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Direction_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0