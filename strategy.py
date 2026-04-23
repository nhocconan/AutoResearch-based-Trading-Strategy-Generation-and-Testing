#!/usr/bin/env python3
"""
Hypothesis: 6-hour price breaks above/below weekly Donchian channels (40-period high/low) with
daily ADX > 25 confirming trend strength. Uses volume confirmation (current volume > 1.5x 20-period average)
to filter breakouts. Designed for low-frequency, high-conviction trades in both bull and bear markets
by requiring multiple confluence factors. Targets 15-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for weekly Donchian and ADX - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate weekly Donchian channels (40-period ≈ 8 weeks of daily data)
    highest_high_40 = pd.Series(high_1d).rolling(window=40, min_periods=40).max()
    lowest_low_40 = pd.Series(low_1d).rolling(window=40, min_periods=40).min()
    donchian_high = highest_high_40.values
    donchian_low = lowest_low_40.values
    
    # Calculate daily ADX (14-period)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
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
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean()
    vol_ma_values = vol_ma_20.values
    
    # Align HTF indicators to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current 6h volume > 1.5x 20-day average volume (scaled)
        # Scale daily volume to 6h approximation: daily volume / 4 (since 4x 6h in a day)
        vol_ma_6h_scaled = vol_ma_aligned[i] / 4.0
        volume_ok = volume[i] > 1.5 * vol_ma_6h_scaled
        
        if position == 0:
            # Long: price breaks above weekly Donchian high + ADX > 25 + volume confirmation
            if (close[i] > donchian_high_aligned[i] and 
                adx_aligned[i] > 25 and 
                volume_ok):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low + ADX > 25 + volume confirmation
            elif (close[i] < donchian_low_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  volume_ok):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to opposite Donchian level or ADX weakens
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below weekly Donchian low OR ADX < 20
                if (close[i] < donchian_low_aligned[i] or adx_aligned[i] < 20):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above weekly Donchian high OR ADX < 20
                if (close[i] > donchian_high_aligned[i] or adx_aligned[i] < 20):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WeeklyDonchian_Breakout_ADX_Volume"
timeframe = "6h"
leverage = 1.0