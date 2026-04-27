#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Direction_VolumeConfirm
Hypothesis: 6h Donchian(20) breakout in direction of weekly pivot trend with volume confirmation.
Weekly pivot trend: price above/below weekly pivot point (PP). Long when price breaks above Donchian upper AND price > weekly PP AND volume spike.
Short when price breaks below Donchian lower AND price < weekly PP AND volume spike.
Weekly pivot provides structural bias; Donchian breakout gives entry timing; volume confirms conviction.
Designed for 12-30 trades/year on 6h to minimize fee drag while working in both bull and bear markets via pivot filtering.
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
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    # Calculate weekly pivot point: (H + L + C) / 3
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    weekly_pp = typical_price.values
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position size
    
    # Warmup: need enough for Donchian(20), weekly PP aligned, volume average
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(weekly_pp_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        weekly_pivot = weekly_pp_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Flat - look for Donchian breakout in direction of weekly pivot trend with volume spike
            # Long: price > Donchian upper AND price > weekly PP AND volume spike
            # Short: price < Donchian lower AND price < weekly PP AND volume spike
            long_condition = (close_val > donchian_high[i]) and (close_val > weekly_pivot) and vol_spike
            short_condition = (close_val < donchian_low[i]) and (close_val < weekly_pivot) and vol_spike
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price breaks below Donchian lower (breakdown) OR weekly trend turns down
            if close_val < donchian_low[i] or close_val < weekly_pivot:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above Donchian upper (breakout) OR weekly trend turns up
            if close_val > donchian_high[i] or close_val > weekly_pivot:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Direction_VolumeConfirm"
timeframe = "6h"
leverage = 1.0