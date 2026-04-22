#!/usr/bin/env python3
"""
12h Donchian(20) breakout with 1d ADX trend filter and volume spike.
Long: price breaks above Donchian(20) high + ADX > 25 + volume spike > 1.5x avg volume.
Short: price breaks below Donchian(20) low + ADX > 25 + volume spike > 1.5x avg volume.
Exit: price crosses Donchian midpoint or ADX < 20.
Breakout provides clear entry, ADX ensures trending market, volume confirms institutional participation.
Works in both bull and bear markets by capturing strong trends filtered by ADX.
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
    
    # Load 1-day data for ADX and volume filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_ma = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_ma = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_ma = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_ma / tr_ma
    di_minus = 100 * dm_minus_ma / tr_ma
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # 1-day average volume
    volume_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h timeframe
    adx_12h = align_htf_to_ltf(prices, df_1d, adx)
    avg_vol_1d_12h = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Calculate Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(adx_12h[i]) or np.isnan(avg_vol_1d_12h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > (1.5 * avg_vol_1d_12h[i])
        
        if position == 0:
            # Long: breakout above Donchian high with trend and volume confirmation
            if close[i] > donch_high[i] and adx_12h[i] > 25 and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low with trend and volume confirmation
            elif close[i] < donch_low[i] and adx_12h[i] > 25 and volume_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below midpoint or ADX weakens
                if close[i] < donch_mid[i] or adx_12h[i] < 20:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above midpoint or ADX weakens
                if close[i] > donch_mid[i] or adx_12h[i] < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_ADX25_VolumeSpike"
timeframe = "12h"
leverage = 1.0