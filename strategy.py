#!/usr/bin/env python3
"""
Hypothesis: 6-hour Donchian(20) breakout with 1-day ADX trend filter and volume confirmation.
Long when price breaks above Donchian upper band, ADX > 25 (trending), and volume > 1.5x 20-period average.
Short when price breaks below Donchian lower band, ADX > 25 (trending), and volume > 1.5x 20-period average.
Exit when price crosses back through Donchian middle (20-period average) or ADX < 20.
Designed for moderate trade frequency (~15-30/year) to capture strong trends while avoiding whipsaws in ranging markets.
Works in both bull and bear markets by requiring strong trend (ADX>25) and volume confirmation.
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
    
    # Calculate Donchian channels (20-period) on 6h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    donchian_middle = ((highest_high + lowest_low) / 2).values
    
    # Load 1-day data for ADX - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
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
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        vol_ratio = volume[i] / vol_avg[i] if vol_avg[i] > 0 else 0
        
        if position == 0:
            # Long: breakout above upper band, ADX > 25, volume > 1.5x average
            if (close[i] > donchian_upper[i] and 
                adx_val > 25 and 
                vol_ratio > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower band, ADX > 25, volume > 1.5x average
            elif (close[i] < donchian_lower[i] and 
                  adx_val > 25 and 
                  vol_ratio > 1.5):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below middle OR ADX < 20 (weakening trend)
                if (close[i] < donchian_middle[i] or adx_val < 20):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above middle OR ADX < 20 (weakening trend)
                if (close[i] > donchian_middle[i] or adx_val < 20):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_1dADX_Volume_Breakout"
timeframe = "6h"
leverage = 1.0