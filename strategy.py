#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Donchian(20) Breakout + Volume Spike + ADX Filter ===
    # Get 12h data
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Breakout signals
    breakout_up = close_12h > donch_high
    breakout_down = close_12h < donch_low
    
    # Volume confirmation (12h volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_12h > vol_ma_20 * 1.5
    
    # ADX trend filter (1-day)
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
    
    # Align indicators to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    breakout_up_aligned = align_htf_to_ltf(prices, df_12h, breakout_up)
    breakout_down_aligned = align_htf_to_ltf(prices, df_12h, breakout_down)
    vol_spike_aligned = align_htf_to_ltf(prices, df_12h, vol_spike)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(breakout_up_aligned[i]) or np.isnan(breakout_down_aligned[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Trend filter: only trade when ADX > 25 (trending market)
        trending = adx_aligned[i] > 25
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Donchian breakout up + volume spike + trending
            if breakout_up_aligned[i] and vol_spike_aligned[i] and trending:
                signals[i] = 0.25
                position = 1
                continue
            # Short: Donchian breakout down + volume spike + trending
            elif breakout_down_aligned[i] and vol_spike_aligned[i] and trending:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long when price touches opposite Donchian band or trend weakens
            if close[i] <= donch_low_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price touches opposite Donchian band or trend weakens
            if close[i] >= donch_high_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dADX25_Volume1.5x"
timeframe = "12h"
leverage = 1.0