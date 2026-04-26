#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirmation_v1
Hypothesis: 6h Donchian(20) breakout traded in direction of weekly pivot trend (price above/below weekly pivot) with volume confirmation (1.5x average). Weekly pivot acts as regime filter: long only when price > weekly pivot, short only when price < weekly pivot. This reduces false breakouts in choppy markets. Target 12-30 trades/year to stay within fee-efficient range. Works in bull markets (breakouts with trend) and bear markets (avoids false longs in downtrend, false shorts in uptrend).
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
    
    # Pre-compute session filter (UTC 8-20) for institutional activity
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot point (P) = (H + L + C) / 3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    if len(high_1w) < 2:
        weekly_pivot = np.full_like(close_1w, np.nan)
    else:
        weekly_pivot = (high_1w[:-1] + low_1w[:-1] + close_1w[:-1]) / 3.0
        weekly_pivot = np.concatenate([[np.nan], weekly_pivot])
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Get 1d data for Donchian calculation (more stable than 6h for channel)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Donchian(20) on daily: upper = max(high, 20), lower = min(low, 20)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    if len(high_1d) < 20:
        donchian_upper = np.full_like(high_1d, np.nan)
        donchian_lower = np.full_like(low_1d, np.nan)
    else:
        # Rolling window of 20 on daily data
        high_series = pd.Series(high_1d)
        low_series = pd.Series(low_1d)
        donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
        donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
        # Pad beginning with NaN
        donchian_upper = np.concatenate([np.full(19, np.nan), donchian_upper[19:]])
        donchian_lower = np.concatenate([np.full(19, np.nan), donchian_lower[19:]])
    
    # Align Donchian levels to 6h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Volume average (20-period = ~5 days on 6h) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(20, 20)  # Donchian, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma[i]) or
            not in_session[i]):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        pivot_val = weekly_pivot_aligned[i]
        upper_val = donchian_upper_aligned[i]
        lower_val = donchian_lower_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = vol_val > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: price breaks above Donchian upper with volume and price > weekly pivot
            long_signal = (close_val > upper_val) and (close_val > pivot_val) and volume_confirmed
            # Short: price breaks below Donchian lower with volume and price < weekly pivot
            short_signal = (close_val < lower_val) and (close_val < pivot_val) and volume_confirmed
            
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
            # Exit: price breaks below Donchian lower (failed breakout/reversal)
            if close_val < lower_val:
                signals[i] = 0.0
                position = 0
            # Exit: price crosses below weekly pivot (regime change)
            elif close_val < pivot_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above Donchian upper (failed breakout/reversal)
            if close_val > upper_val:
                signals[i] = 0.0
                position = 0
            # Exit: price crosses above weekly pivot (regime change)
            elif close_val > pivot_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirmation_v1"
timeframe = "6h"
leverage = 1.0