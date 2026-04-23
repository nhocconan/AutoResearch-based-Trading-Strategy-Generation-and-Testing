#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian breakout (20-period) with volume confirmation and ADX(20) trend filter.
Long when price breaks above Donchian high, volume > 1.5x average, and ADX > 20.
Short when price breaks below Donchian low, volume > 1.5x average, and ADX > 20.
Exit when price reverses to Donchian midpoint or ADX < 15.
Designed for ~20-40 trades/year per symbol to capture strong trends while minimizing whipsaws.
Works in both bull and bear markets by requiring trend confirmation (ADX > 20) and volume spikes.
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
    
    # Load 1-day data for ADX - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day ADX (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr20 = pd.Series(tr).rolling(window=20, min_periods=20).sum()
    dm_plus20 = pd.Series(dm_plus).rolling(window=20, min_periods=20).sum()
    dm_minus20 = pd.Series(dm_minus).rolling(window=20, min_periods=20).sum()
    
    # Directional Indicators
    plus_di = 100 * dm_plus20 / tr20
    minus_di = 100 * dm_minus20 / tr20
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=20, min_periods=20).mean()
    adx_values = adx.values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Calculate Donchian channels (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume average (20-period) on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high, volume confirmation, ADX > 20
            if (close[i] > donchian_high[i] and 
                vol_current > 1.5 * vol_ma_val and 
                adx_val > 20):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low, volume confirmation, ADX > 20
            elif (close[i] < donchian_low[i] and 
                  vol_current > 1.5 * vol_ma_val and 
                  adx_val > 20):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to Donchian midpoint OR ADX < 15 (trend weakening)
                if (close[i] <= donchian_mid[i]) or adx_val < 15:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to Donchian midpoint OR ADX < 15 (trend weakening)
                if (close[i] >= donchian_mid[i]) or adx_val < 15:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dADX_Volume_Breakout"
timeframe = "4h"
leverage = 1.0