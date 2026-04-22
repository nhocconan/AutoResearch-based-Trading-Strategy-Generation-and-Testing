#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian channel breakout with 1-day volume confirmation and 1-week ADX trend filter.
Long when price breaks above Donchian(20) high, 1-day volume > 1.5x 20-day average, and 1-week ADX > 25.
Short when price breaks below Donchian(20) low, 1-day volume > 1.5x 20-day average, and 1-week ADX > 25.
Exit when price crosses opposite Donchian boundary or ADX falls below 20.
Donchian provides clear breakout levels; volume confirms institutional interest; ADX filters for trending markets.
Designed for low trade frequency by requiring multiple confirmations and using higher timeframe filters.
Works in both bull and bear markets by only taking trades when ADX confirms a trend is present.
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
    
    # Load 1-day data for volume average - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1-day volume 20-day average
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Load 1-week data for ADX - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX(14) on 1-week data
    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First value has no previous close
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    # First values have no previous
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_14 / tr14
    minus_di = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Donchian channel (20-period) on 12-hour data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high, volume confirmation, ADX > 25
            if (close[i] > high_20[i] and 
                volume[i] > 1.5 * vol_ma_20_aligned[i] and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low, volume confirmation, ADX > 25
            elif (close[i] < low_20[i] and 
                  volume[i] > 1.5 * vol_ma_20_aligned[i] and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below Donchian low OR ADX falls below 20
                if (close[i] < low_20[i] or 
                    adx_aligned[i] < 20):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above Donchian high OR ADX falls below 20
                if (close[i] > high_20[i] or 
                    adx_aligned[i] < 20):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_1dVolume_1wADX_Trend"
timeframe = "12h"
leverage = 1.0