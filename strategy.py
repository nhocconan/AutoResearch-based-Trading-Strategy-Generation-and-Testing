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
    
    # === Weekly Donchian channel (20-period) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian to daily
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    
    # === Daily ADX(14) for trend filter ===
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
    
    # Align 1d ADX to daily
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === Daily volume confirmation ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 30
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_20_aligned[i-1]  # Break above previous period's high
        breakout_down = close[i] < low_20_aligned[i-1]  # Break below previous period's low
        
        # Trend filter
        trending = adx_1d_aligned[i] > 25
        
        # Volume spike: current daily volume > 1.5x 20-period average
        vol_spike = volume[i] > vol_ma_20_1d_aligned[i] * 1.5
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Donchian breakout up + volume spike + trending
            if breakout_up and vol_spike and trending:
                signals[i] = 0.25
                position = 1
                continue
            # Short: Donchian breakout down + volume spike + trending
            elif breakout_down and vol_spike and trending:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long when price returns to middle of channel or trend weakens
            mid_channel = (high_20_aligned[i] + low_20_aligned[i]) / 2
            if close[i] < mid_channel or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price returns to middle of channel or trend weakens
            mid_channel = (high_20_aligned[i] + low_20_aligned[i]) / 2
            if close[i] > mid_channel or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian20_1dADX25_Volume1.5x"
timeframe = "1d"
leverage = 1.0