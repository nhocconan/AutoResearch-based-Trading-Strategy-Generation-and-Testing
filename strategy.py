#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 4h Donchian channel (20-period) ===
    high_20 = pd.Series(close).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(close).rolling(window=20, min_periods=20).min().values
    
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
    plus_di = 100 * dm_plus_14 / (tr14 + 1e-10)
    minus_di = 100 * dm_minus_14 / (tr14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 4h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 1d volume confirmation ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === 4h volume confirmation ===
    vol_ma_10_4h = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 30
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(vol_ma_10_4h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 1d volume
        df_1d_current = get_htf_data(prices, '1d')
        volume_1d_current = df_1d_current['volume'].values
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d_current, volume_1d_current)
        
        # Volume spike: current 1d volume > 1.5x 20-period average AND 4h volume > 1.3x 10-period average
        vol_spike_1d = volume_1d_aligned[i] > vol_ma_20_1d_aligned[i] * 1.5
        vol_spike_4h = volume[i] > vol_ma_10_4h[i] * 1.3
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_20[i-1]  # Break above previous period's high
        breakout_down = close[i] < low_20[i-1]  # Break below previous period's low
        
        # Trend filter
        trending = adx_1d_aligned[i] > 25
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Donchian breakout up + volume spike + trending
            if breakout_up and vol_spike_1d and vol_spike_4h and trending:
                signals[i] = 0.25
                position = 1
                continue
            # Short: Donchian breakout down + volume spike + trending
            elif breakout_down and vol_spike_1d and vol_spike_4h and trending:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long when price returns to middle of channel or trend weakens
            mid_channel = (high_20[i] + low_20[i]) / 2
            if close[i] < mid_channel or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price returns to middle of channel or trend weakens
            mid_channel = (high_20[i] + low_20[i]) / 2
            if close[i] > mid_channel or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dADX25_Volume1.5x_4hVol1.3x"
timeframe = "4h"
leverage = 1.0