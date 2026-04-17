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
    
    # === 4h price channel: 20-period Donchian breakout ===
    # Calculate high and low of last 20 periods
    high_20 = pd.Series(close).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(close).rolling(window=20, min_periods=20).min().values
    
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
    
    # Align 12h ADX to 4h
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # === 12h volume confirmation ===
    volume_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
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
            np.isnan(adx_12h_aligned[i]) or 
            np.isnan(vol_ma_20_12h_aligned[i]) or np.isnan(vol_ma_10_4h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 12h volume
        df_12h_current = get_htf_data(prices, '12h')
        volume_12h_current = df_12h_current['volume'].values
        volume_12h_aligned = align_htf_to_ltf(prices, df_12h_current, volume_12h_current)
        
        # Volume spike: current 12h volume > 1.5x 20-period average AND 4h volume > 1.3x 10-period average
        vol_spike_12h = volume_12h_aligned[i] > vol_ma_20_12h_aligned[i] * 1.5
        vol_spike_4h = volume[i] > vol_ma_10_4h[i] * 1.3
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_20[i-1]  # Break above previous period's high
        breakout_down = close[i] < low_20[i-1]  # Break below previous period's low
        
        # Trend filter
        trending = adx_12h_aligned[i] > 25
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Donchian breakout up + volume spike + trending
            if breakout_up and vol_spike_12h and vol_spike_4h and trending:
                signals[i] = 0.25
                position = 1
                continue
            # Short: Donchian breakout down + volume spike + trending
            elif breakout_down and vol_spike_12h and vol_spike_4h and trending:
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

name = "4h_Donchian20_12hADX25_Volume1.5x_4hVol1.3x"
timeframe = "4h"
leverage = 1.0