#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Spike and ADX Trend Filter.
Trades breakouts above/below Donchian channels (20-period) confirmed by volume spikes,
only in trending markets (1-day ADX > 25) to avoid false breakouts in ranging conditions.
Designed for 4h timeframe to target 75-200 total trades over 4 years (19-50/year).
Works in both bull and bear markets by following the trend direction.
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
    
    # Get 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian Channel (20-period)
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Breakout conditions
    breakout_up = high_4h > upper_20
    breakout_down = low_4h < lower_20
    
    # Align signals to 4h timeframe
    breakout_up_aligned = align_htf_to_ltf(prices, df_4h, breakout_up.astype(float))
    breakout_down_aligned = align_htf_to_ltf(prices, df_4h, breakout_down.astype(float))
    
    # Get 1d data for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Volume spike: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 1.8)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # 1-day ADX (14-period) for trend filter
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr0 = np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])
    tr = np.concatenate([[tr0], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    # Handle division by zero
    dx = np.where((di_plus + di_minus) > 0, dx, 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # ADX > 25 = trending market
    trending = adx > 25
    trending_aligned = align_htf_to_ltf(prices, df_1d, trending.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(breakout_up_aligned[i]) or 
            np.isnan(breakout_down_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(trending_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Donchian breakout + volume spike + trending market
        long_entry = (breakout_up_aligned[i] > 0.5 and 
                      vol_spike_aligned[i] > 0.5 and 
                      trending_aligned[i] > 0.5)
        short_entry = (breakout_down_aligned[i] > 0.5 and 
                       vol_spike_aligned[i] > 0.5 and 
                       trending_aligned[i] > 0.5)
        
        # Exit when price returns to middle of Donchian channel
        middle = (upper_20 + lower_20) / 2
        middle_aligned = align_htf_to_ltf(prices, df_4h, np.full_like(close_4h, middle))
        
        exit_long = position == 1 and close[i] <= middle_aligned[i]
        exit_short = position == -1 and close[i] >= middle_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_donchian_breakout_volume_trend"
timeframe = "4h"
leverage = 1.0