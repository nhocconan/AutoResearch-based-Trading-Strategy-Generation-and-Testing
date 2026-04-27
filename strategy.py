#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_VolumeConfirm
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
Weekly pivot (from 1w data) determines bias: long only above weekly pivot, short only below.
Donchian breakout provides entry timing, volume confirms conviction.
Designed for 12-30 trades/year on 6h to minimize fee drag while capturing trending moves.
Works in both bull and bear markets by using weekly pivot as regime filter.
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
    
    # Load 1d data ONCE before loop for EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Load 1w data ONCE before loop for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    # Weekly pivot: (H+L+C)/3 from previous completed weekly bar
    weekly_pivot = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25  # 25% position size
    
    # Warmup: need enough for Donchian(20), EMA50, volume average
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        ema_trend = ema_50_1d_aligned[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Flat - look for Donchian breakout with weekly pivot bias and volume spike
            # Long: price breaks above Donchian HIGH AND above weekly pivot AND volume spike
            # Short: price breaks below Donchian LOW AND below weekly pivot AND volume spike
            long_condition = (close_val > donch_high) and (close_val > weekly_pivot_val) and vol_spike
            short_condition = (close_val < donch_low) and (close_val < weekly_pivot_val) and vol_spike
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Long - exit when price breaks below Donchian LOW OR weekly pivot
            if close_val < donch_low or close_val < weekly_pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above Donchian HIGH OR weekly pivot
            if close_val > donch_high or close_val > weekly_pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_WeeklyPivot_VolumeConfirm"
timeframe = "6h"
leverage = 1.0