#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + Volume Spike + ADX Trend Filter (1d)
Long when price breaks above upper Donchian band with volume > 1.5x average and ADX > 25.
Short when price breaks below lower Donchian band with volume > 1.5x average and ADX > 25.
Exit when price crosses the middle Donchian band or ADX < 20.
Designed for low turnover: ~20-30 trades/year per symbol.
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
    
    # Load 1-day data once for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14)
    adx_period = 14
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    
    # Directional indicators
    plus_di = 100 * dm_plus_smooth / atr
    minus_di = 100 * dm_minus_smooth / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    
    # Donchian channels (20)
    donch_period = 20
    upper = pd.Series(high).rolling(window=donch_period, min_periods=donch_period).max().values
    lower = pd.Series(low).rolling(window=donch_period, min_periods=donch_period).min().values
    middle = (upper + lower) / 2
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(30, n):
        # 1-day index (6 bars per day for 4h timeframe)
        idx_1d = i // 6
        if idx_1d < 1:
            continue
        
        # Use previous 1d values to avoid look-ahead
        prev_idx = idx_1d - 1
        if prev_idx < 0:
            continue
            
        # Get values from previous 1d bar
        adx_prev = adx[prev_idx] if prev_idx < len(adx) else adx[-1]
        
        if np.isnan(adx_prev):
            continue
        
        # Create array for alignment
        adx_arr = np.full(len(df_1d), adx_prev)
        adx_4h = align_htf_to_ltf(prices, df_1d, adx_arr)[i]
        
        if np.isnan(adx_4h):
            continue
        
        if position == 0:
            # Long: Break above upper Donchian with volume spike and ADX > 25 (trending)
            if close[i] > upper[i] and volume[i] > vol_ma[i] * 1.5 and adx_4h > 25:
                position = 1
                signals[i] = position_size
            # Short: Break below lower Donchian with volume spike and ADX > 25 (trending)
            elif close[i] < lower[i] and volume[i] > vol_ma[i] * 1.5 and adx_4h > 25:
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Cross below middle Donchian or ADX < 20 (weak trend)
            if close[i] < middle[i] or adx_4h < 20:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Cross above middle Donchian or ADX < 20 (weak trend)
            if close[i] > middle[i] or adx_4h < 20:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0