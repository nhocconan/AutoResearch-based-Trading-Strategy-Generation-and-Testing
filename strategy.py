#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R with 1d volume confirmation and ADX trend filter.
- Williams %R(14) identifies overbought/oversold conditions
- Only trade when 1d ADX > 25 (trending market) to avoid chop
- Require 1d volume > 1.5x 20-period average for confirmation
- Enter long when %R crosses above -80 from below (oversold bounce)
- Enter short when %R crosses below -20 from above (overbought rejection)
- Exit on opposite signal or when trend weakens (ADX < 20)
- Designed for 4h timeframe to capture swings in both bull and bear markets
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
    
    # === Williams %R(14) on 4h ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # === 1d ADX(25) for trend filter ===
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
    tr25 = pd.Series(tr).rolling(window=25, min_periods=25).sum().values
    dm_plus_25 = pd.Series(dm_plus).rolling(window=25, min_periods=25).sum().values
    dm_minus_25 = pd.Series(dm_minus).rolling(window=25, min_periods=25).sum().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_25 / tr25
    minus_di = 100 * dm_minus_25 / tr25
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=25, min_periods=25).mean().values
    
    # Align 1d ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 1d volume confirmation ===
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 40
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 1d volume and align
        df_1d_current = get_htf_data(prices, '1d')
        vol_1d_current = df_1d_current['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d_current, vol_1d_current)
        
        # Volume spike: current 1d volume > 1.5x 20-period average
        vol_spike = vol_1d_aligned[i] > vol_ma_20_aligned[i] * 1.5
        
        # Williams %R conditions
        wr = williams_r[i]
        wr_prev = williams_r[i-1]
        
        # Bullish: %R crosses above -80 from below (oversold bounce)
        bullish_cross = (wr_prev <= -80) and (wr > -80)
        # Bearish: %R crosses below -20 from above (overbought rejection)
        bearish_cross = (wr_prev >= -20) and (wr < -20)
        
        # ADX trend filter
        trending = adx_aligned[i] > 25
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: bullish cross + volume spike + trending
            if bullish_cross and vol_spike and trending:
                signals[i] = 0.25
                position = 1
                continue
            # Short: bearish cross + volume spike + trending
            elif bearish_cross and vol_spike and trending:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long on bearish cross or trend weakness
            if bearish_cross or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short on bullish cross or trend weakness
            if bullish_cross or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR14_1dVolume1.5x_1dADX25"
timeframe = "4h"
leverage = 1.0