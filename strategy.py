#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian breakout with 1-day volume confirmation and 1-day ADX trend filter.
Long when price breaks above Donchian upper band (20) and 1-day volume > average + ADX > 25.
Short when price breaks below Donchian lower band (20) and 1-day volume > average + ADX > 25.
Exit when price crosses Donchian middle line or volume drops below average.
Donchian provides clear breakout signals; volume confirms institutional interest; ADX filters non-trending markets.
Works in trending markets by capturing breakouts and avoids ranging markets via ADX filter.
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
    
    # Load 1-day data for volume and ADX filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1-day volume average
    volume_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # 1-day ADX calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus14 / tr14
    di_minus = 100 * dm_minus14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(avg_vol_1d_aligned[i]) or np.isnan(adx_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high, volume above average, ADX > 25
            if close[i] > donch_high[i] and volume_1d[i] > avg_vol_1d_aligned[i] and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low, volume above average, ADX > 25
            elif close[i] < donch_low[i] and volume_1d[i] > avg_vol_1d_aligned[i] and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below Donchian mid
                if close[i] < donch_mid[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above Donchian mid
                if close[i] > donch_mid[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_Breakout_1dVolume_ADX_Filter"
timeframe = "4h"
leverage = 1.0