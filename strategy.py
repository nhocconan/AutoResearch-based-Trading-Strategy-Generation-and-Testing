#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeSpike
Hypothesis: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot direction (price above/below weekly pivot) and volume spikes (>2x 20-bar avg) capture institutional breakouts with reduced false signals. Weekly pivot provides structural bias from higher timeframe, reducing whipsaws in both bull and bear markets. Targets 15-30 trades/year to minimize fee drag.
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
    
    # Get 1w data for weekly pivot (primary HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    # P = (H + L + C) / 3
    weekly_pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    weekly_pivot_vals = weekly_pivot.values
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_vals)
    
    # Get 1d data for additional trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian(20) on 6h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume spike: >2x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for all calculations
    start_idx = max(lookback, 60, 20)  # Donchian20, 1d EMA50, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get values
        weekly_pivot_val = weekly_pivot_aligned[i]
        ema_50_val = ema_50_aligned[i]
        highest_high_val = highest_high[i]
        lowest_low_val = lowest_low[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume confirmation: current volume > 2x 20-period average
        volume_confirmed = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: Donchian breakout above weekly pivot and EMA50 with volume
            long_signal = (high_val > highest_high_val) and (close_val > weekly_pivot_val) and (close_val > ema_50_val) and volume_confirmed
            # Short: Donchian breakdown below weekly pivot and EMA50 with volume
            short_signal = (low_val < lowest_low_val) and (close_val < weekly_pivot_val) and (close_val < ema_50_val) and volume_confirmed
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below weekly pivot OR Donchian lower band
            if close_val < weekly_pivot_val or close_val < lowest_low_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above weekly pivot OR Donchian upper band
            if close_val > weekly_pivot_val or close_val > highest_high_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeSpike"
timeframe = "6h"
leverage = 1.0