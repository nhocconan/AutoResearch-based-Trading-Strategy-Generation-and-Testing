#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(10) breakout with 1w volume confirmation and 1d ADX trend filter.
- Donchian(10) breakout on 12h provides clear entry/exit with lower frequency
- Only trade when 1w volume > 1.3x 50-period average for institutional participation
- Require 1d ADX > 20 (trending market) to avoid chop and false breakouts
- Enter long when price breaks above 10-period high
- Enter short when price breaks below 10-period low
- Exit when price returns to 5-period mid-band or ADX weakens
- Designed for 12h timeframe to capture major trends with minimal trades
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === Donchian channels on 12h (10-period) ===
    high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    mid_5 = (pd.Series(high).rolling(window=5, min_periods=5).max().values + 
             pd.Series(low).rolling(window=5, min_periods=5).min().values) / 2
    
    # === 1w volume confirmation ===
    df_1w = get_htf_data(prices, '1w')
    volume_1w = df_1w['volume'].values
    vol_ma_50 = pd.Series(volume_1w).rolling(window=50, min_periods=50).mean().values
    vol_ma_50_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_50)
    
    # === 1d ADX(14) for trend filter ===
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
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_14 / tr14
    minus_di = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 12h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(high_10[i]) or np.isnan(low_10[i]) or np.isnan(mid_5[i]) or 
            np.isnan(vol_ma_50_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 1w volume and align
        df_1w_current = get_htf_data(prices, '1w')
        vol_1w_current = df_1w_current['volume'].values
        vol_1w_aligned = align_htf_to_ltf(prices, df_1w_current, vol_1w_current)
        
        # Volume spike: current 1w volume > 1.3x 50-period average
        vol_spike = vol_1w_aligned[i] > vol_ma_50_aligned[i] * 1.3
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_10[i-1]  # Break above previous 10-period high
        breakout_down = close[i] < low_10[i-1]  # Break below previous 10-period low
        return_to_mid = abs(close[i] - mid_5[i]) < (high_10[i] - low_10[i]) * 0.15  # Near mid-band
        
        # ADX trend filter
        trending = adx_aligned[i] > 20
        
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
            if return_to_mid or adx_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price returns to mid-band or trend weakens
            if return_to_mid or adx_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian10_1wVolume1.3x_1dADX20"
timeframe = "12h"
leverage = 1.0