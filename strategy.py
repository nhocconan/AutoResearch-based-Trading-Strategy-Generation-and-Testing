#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # === 1h Donchian breakout (20-period) ===
    high_20 = pd.Series(close).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(close).rolling(window=20, min_periods=20).min().values
    
    # === 4h ADX(14) for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr3 = np.abs(low_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_4h - np.concatenate([[high_4h[0]], high_4h[:-1]])) > 
                       (np.concatenate([[low_4h[0]], low_4h[:-1]]) - low_4h), 
                       np.maximum(high_4h - np.concatenate([[high_4h[0]], high_4h[:-1]]), 0), 0)
    dm_minus = np.where((np.concatenate([[low_4h[0]], low_4h[:-1]]) - low_4h) > 
                        (high_4h - np.concatenate([[high_4h[0]], high_4h[:-1]])), 
                        np.maximum(np.concatenate([[low_4h[0]], low_4h[:-1]]) - low_4h, 0), 0)
    
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
    
    # Align 4h ADX to 1h
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # === 4h volume confirmation ===
    volume_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    # === 1h volume confirmation ===
    vol_ma_10_1h = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    # Session filter: 08-20 UTC (precomputed for efficiency)
    hours = pd.DatetimeIndex(open_time).hour
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 30
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(adx_4h_aligned[i]) or 
            np.isnan(vol_ma_20_4h_aligned[i]) or np.isnan(vol_ma_10_1h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume spike: current 4h volume > 1.5x 20-period average AND 1h volume > 1.3x 10-period average
        vol_spike_4h = volume[i] > vol_ma_20_4h_aligned[i] * 1.5
        vol_spike_1h = volume[i] > vol_ma_10_1h[i] * 1.3
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_20[i-1]  # Break above previous period's high
        breakout_down = close[i] < low_20[i-1]  # Break below previous period's low
        
        # Trend filter
        trending = adx_4h_aligned[i] > 25
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Donchian breakout up + volume spike + trending
            if breakout_up and vol_spike_4h and vol_spike_1h and trending:
                signals[i] = 0.20
                position = 1
                continue
            # Short: Donchian breakout down + volume spike + trending
            elif breakout_down and vol_spike_4h and vol_spike_1h and trending:
                signals[i] = -0.20
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long when price returns to middle of channel or trend weakens
            mid_channel = (high_20[i] + low_20[i]) / 2
            if close[i] < mid_channel or adx_4h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short when price returns to middle of channel or trend weakens
            mid_channel = (high_20[i] + low_20[i]) / 2
            if close[i] > mid_channel or adx_4h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_4hADX25_Volume4h1.5x_1h1.3x"
timeframe = "1h"
leverage = 1.0