#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R with 12h ADX trend filter and 1d volume confirmation.
- Williams %R(14): oversold < -80 for long, overbought > -20 for short
- 12h ADX(14) > 25 filters for trending markets only
- 1d volume > 1.5x 20-period average confirms breakout strength
- Targets 20-40 trades/year to avoid fee drain. Works in trending markets with momentum confirmation.
"""

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
    
    # === Williams %R(14) on 6h ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
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
    plus_di = 100 * dm_plus_14 / tr14
    minus_di = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 12h ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # === 1d volume confirmation ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 30
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume spike: current 1d volume > 1.5x 20-period average
        df_1d_current = get_htf_data(prices, '1d')
        vol_1d_current = df_1d_current['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d_current, vol_1d_current)
        vol_spike = vol_1d_aligned[i] > vol_ma_20_aligned[i] * 1.5
        
        # Williams %R conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # ADX trend filter
        trending = adx_aligned[i] > 25
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: oversold + volume spike + trending
            if oversold and vol_spike and trending:
                signals[i] = 0.25
                position = 1
                continue
            # Short: overbought + volume spike + trending
            elif overbought and vol_spike and trending:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal on opposite condition
        elif position == 1:
            # Exit long if overbought or trend weakens
            if overbought or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if oversold or trend weakens
            if oversold or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_12hADX_1dVolume1.5x"
timeframe = "6h"
leverage = 1.0