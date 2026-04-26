#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirmation
Hypothesis: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot direction (from 1w HTF) and volume confirmation capture institutional moves while avoiding whipsaws. 
Weekly pivot provides structural bias: trade breakouts only in direction of weekly pivot (above/below weekly pivot point). 
Volume confirmation ensures breakout legitimacy. Targets 12-30 trades/year to minimize fee drag. 
Works in bull markets (breakouts with trend) and bear markets (mean reversion at extremes during low volatility via pivot filtering).
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
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (UTC 0-24 - trade all hours for 6h)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 0) & (hours <= 23)  # all hours
    
    # Get 1w data for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot point: (H + L + C) / 3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    if len(high_1w) < 1:
        weekly_pivot = np.full_like(close_1w, np.nan)
    else:
        weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Donchian(20) on 6h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume average (24-period = ~6 days on 6h) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(lookback, 24)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(weekly_pivot_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get values
        hh_val = highest_high[i]
        ll_val = lowest_low[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 24-period average
        volume_confirmed = vol_val > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: price breaks above Donchian high AND above weekly pivot with volume confirmation
            long_signal = (high_val > hh_val) and (close_val > weekly_pivot_val) and volume_confirmed
            # Short: price breaks below Donchian low AND below weekly pivot with volume confirmation
            short_signal = (low_val < ll_val) and (close_val < weekly_pivot_val) and volume_confirmed
            
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
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks below Donchian low
            if low_val < ll_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Price crosses below weekly pivot (trend change)
            elif close_val < weekly_pivot_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks above Donchian high
            if high_val > hh_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Price crosses above weekly pivot (trend change)
            elif close_val > weekly_pivot_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0