#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Donchian channel (20-period) ===
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h ADX(14) for trend filter ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr3 = np.abs(low_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.concatenate([[high_12h[0]], high_12h[:-1]])) > 
                       (np.concatenate([[low_12h[0]], low_12h[:-1]]) - low_12h), 
                       np.maximum(high_12h - np.concatenate([[high_12h[0]], high_12h[:-1]]), 0), 0)
    dm_minus = np.where((np.concatenate([[low_12h[0]], low_12h[:-1]]) - low_12h) > 
                        (high_12h - np.concatenate([[high_12h[0]], high_12h[:-1]])), 
                        np.maximum(np.concatenate([[low_12h[0]], low_12h[:-1]]) - low_12h, 0), 0)
    
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
    
    # Align 12h ADX to 6h
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # === 12h volume confirmation ===
    volume_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # === 6h volume confirmation ===
    vol_ma_10_6h = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 30
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(adx_12h_aligned[i]) or 
            np.isnan(vol_ma_20_12h_aligned[i]) or np.isnan(vol_ma_10_6h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_20[i-1]  # Break above previous period's high
        breakout_down = close[i] < low_20[i-1]  # Break below previous period's low
        
        # Trend filter
        trending = adx_12h_aligned[i] > 25
        
        # Volume spike: current 12h volume > 1.5x 20-period average AND 6h volume > 1.3x 10-period average
        vol_spike_12h = volume_12h[i // 2] > vol_ma_20_12h[i // 2] * 1.5 if i >= 2 else False
        vol_spike_6h = volume[i] > vol_ma_10_6h[i] * 1.3
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Donchian breakout up + volume spike + trending
            if breakout_up and vol_spike_12h and vol_spike_6h and trending:
                signals[i] = 0.25
                position = 1
                continue
            # Short: Donchian breakout down + volume spike + trending
            elif breakout_down and vol_spike_12h and vol_spike_6h and trending:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long when price returns to middle of channel or trend weakens
            mid_channel = (high_20[i] + low_20[i]) / 2
            if close[i] < mid_channel or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price returns to middle of channel or trend weakens
            mid_channel = (high_20[i] + low_20[i]) / 2
            if close[i] > mid_channel or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_12hADX25_Volume1.5x_6hVol1.3x"
timeframe = "6h"
leverage = 1.0