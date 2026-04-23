#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian breakout with 1-day ADX trend filter and volume confirmation.
Long when price breaks above Donchian upper channel (20) AND daily ADX > 25 (trending) AND volume > 1.5x average volume.
Short when price breaks below Donchian lower channel (20) AND daily ADX > 25 (trending) AND volume > 1.5x average volume.
Exit when price crosses the Donchian middle line (average of upper and lower) or ADX drops below 20.
Designed for low-moderate trade frequency (~20-40/year) to capture strong trends while avoiding whipsaws in ranging markets.
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
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    upper_channel = highest_high.values
    lower_channel = lowest_low.values
    middle_channel = (upper_channel + lower_channel) / 2
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(avg_volume[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        vol_ratio = volume[i] / avg_volume[i] if avg_volume[i] > 0 else 0
        
        if position == 0:
            # Long: price breaks above upper channel AND ADX > 25 AND volume > 1.5x average
            if (close[i] > upper_channel[i] and 
                adx_val > 25 and 
                vol_ratio > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel AND ADX > 25 AND volume > 1.5x average
            elif (close[i] < lower_channel[i] and 
                  adx_val > 25 and 
                  vol_ratio > 1.5):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below middle line OR ADX < 20 (weakening trend)
                if close[i] < middle_channel[i] or adx_val < 20:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above middle line OR ADX < 20 (weakening trend)
                if close[i] > middle_channel[i] or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_Breakout_1dADX_Volume"
timeframe = "4h"
leverage = 1.0