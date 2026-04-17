#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 1d volume confirmation and ADX trend filter.
- Donchian(20) breakout provides clear entry/exit signals
- Only trade when 1d ADX > 25 (trending market) to avoid chop
- Require 1d volume > 1.5x 20-period average for confirmation
- Enter long when price breaks above 20-period high
- Enter short when price breaks below 20-period low
- Exit when price returns to 10-period mid-band
- Designed for 4h timeframe to capture trends in both bull and bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === Donchian channels on 4h ===
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_10 = (pd.Series(high).rolling(window=10, min_periods=10).max().values + 
              pd.Series(low).rolling(window=10, min_periods=10).min().values) / 2
    
    # === 1d ADX(25) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])) > 
                       (np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d), 
                       np.maximum(high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]]), 0), 0)
    dm_minus = np.where((np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d) > 
                        (high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])), 
                        np.maximum(np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d, 0), 0)
    
    # Smoothed values
    tr25 = pd.Series(tr).rolling(window=25, min_periods=25).sum().values
    dm_plus_25 = pd.Series(dm_plus).rolling(window=25, min_periods=25).sum().values
    dm_minus_25 = pd.Series(dm_minus).rolling(window=25, min_periods=25).sum().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_25 / tr25
    minus_di = 100 * dm_minus_25 / tr25
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=25, min_periods=25).mean().values
    
    # Align 1d ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 1d volume confirmation ===
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 25
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(mid_10[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 1d volume and align
        df_1d_current = get_htf_data(prices, '1d')
        vol_1d_current = df_1d_current['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d_current, vol_1d_current)
        
        # Volume spike: current 1d volume > 1.5x 20-period average
        vol_spike = vol_1d_aligned[i] > vol_ma_20_aligned[i] * 1.5
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_20[i-1]  # Break above previous 20-period high
        breakout_down = close[i] < low_20[i-1]  # Break below previous 20-period low
        return_to_mid = abs(close[i] - mid_10[i]) < (high_20[i] - low_20[i]) * 0.1  # Near mid-band
        
        # ADX trend filter
        trending = adx_aligned[i] > 25
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: upward breakout + volume spike + trending
            if breakout_up and vol_spike and trending:
                signals[i] = 0.25
                position = 1
                continue
            # Short: downward breakout + volume spike + trending
            elif breakout_down and vol_spike and trending:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long when price returns to mid-band or trend weakens
            if return_to_mid or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price returns to mid-band or trend weakens
            if return_to_mid or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dVolume1.5x_1dADX25"
timeframe = "4h"
leverage = 1.0