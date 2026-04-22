#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian channel breakout with 1-day ADX trend filter and volume confirmation.
Long when price breaks above Donchian(20) high and ADX > 25 and volume > 20-period average volume.
Short when price breaks below Donchian(20) low and ADX > 25 and volume > 20-period average volume.
Exit when price returns to Donchian midpoint or ADX drops below 20.
Donchian captures breakouts, ADX filters for trending markets, volume confirms institutional interest.
Works in both bull and bear markets by only trading in strong trends (ADX > 25) and using volatility-based stops.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Load 1-day data for ADX filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on daily data
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
    
    # Smooth TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial values (simple average of first 14 periods)
    if len(tr) >= 14:
        atr[13] = np.mean(tr[0:14])
        dm_plus_smooth[13] = np.mean(dm_plus[0:14])
        dm_minus_smooth[13] = np.mean(dm_minus[0:14])
        
        # Wilder's smoothing for subsequent values
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * 13 + dm_plus[i]) / 14
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * 13 + dm_minus[i]) / 14
    
    # Avoid division by zero
    dm_plus_smooth = np.where(dm_plus_smooth == 0, 1e-10, dm_plus_smooth)
    dm_minus_smooth = np.where(dm_minus_smooth == 0, 1e-10, dm_minus_smooth)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    adx = np.zeros_like(dx)
    
    # Initial ADX value (average of first 14 DX values)
    if len(dx) >= 27:  # Need 14 for initial ATR/DM + 14 for ADX
        adx[26] = np.mean(dx[14:28])  # First 14 DX values after stabilization
        
        # Wilder's smoothing for subsequent ADX values
        for i in range(27, len(dx)):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter (20-period average)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(adx_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high, ADX > 25, volume above average
            if close[i] > donchian_high[i] and adx_aligned[i] > 25 and volume[i] > avg_volume[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low, ADX > 25, volume above average
            elif close[i] < donchian_low[i] and adx_aligned[i] > 25 and volume[i] > avg_volume[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to Donchian midpoint OR ADX drops below 20
                if close[i] <= donchian_mid[i] or adx_aligned[i] < 20:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to Donchian midpoint OR ADX drops below 20
                if close[i] >= donchian_mid[i] or adx_aligned[i] < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian_ADX_Volume_Filter"
timeframe = "12h"
leverage = 1.0