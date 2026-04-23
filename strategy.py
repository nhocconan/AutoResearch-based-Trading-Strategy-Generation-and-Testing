#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian channel breakout with 1-day ADX trend filter and volume confirmation.
Long when price breaks above Donchian(20) upper band on 12h, ADX > 25 on 1d, and volume > 1.5x average.
Short when price breaks below Donchian(20) lower band on 12h, ADX > 25 on 1d, and volume > 1.5x average.
Exit when price returns to Donchian midpoint or ADX < 20.
Designed for low frequency (~20-40/year) to capture strong trends with minimal whipsaws.
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
    
    # Load 12-hour data for Donchian calculation - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12-hour Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max()
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min()
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Load 1-day data for ADX and average volume - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day ADX (14-period)
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
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum()
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    plus_di = 100 * dm_plus14 / tr14
    minus_di = 100 * dm_minus14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    
    # Calculate 1-day average volume (20-period)
    volume_1d = df_1d['volume'].values
    avg_volume = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean()
    avg_volume_values = avg_volume.values
    
    # Align HTF indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high.values)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low.values)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid.values)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    avg_volume_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(avg_volume_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high, ADX > 25, volume > 1.5x average
            if (high[i] > donchian_high_aligned[i] and 
                adx_aligned[i] > 25 and 
                volume[i] > 1.5 * avg_volume_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian low, ADX > 25, volume > 1.5x average
            elif (low[i] < donchian_low_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  volume[i] > 1.5 * avg_volume_aligned[i]):
                signals[i] = -0.30
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to Donchian midpoint or ADX < 20
                if (low[i] <= donchian_mid_aligned[i] or adx_aligned[i] < 20):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to Donchian midpoint or ADX < 20
                if (high[i] >= donchian_mid_aligned[i] or adx_aligned[i] < 20):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "12H_Donchian_Breakout_1dADX_Volume"
timeframe = "12h"
leverage = 1.0