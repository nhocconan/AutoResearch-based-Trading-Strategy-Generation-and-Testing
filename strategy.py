#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout (20-period) with 1d ADX trend filter and volume confirmation.
# Donchian breakouts capture momentum in trending markets; ADX > 25 ensures we only trade strong trends.
# Volume > 1.5x 20-period average confirms breakout validity.
# Works in both bull and bear markets by going long on upper breakouts and short on lower breakouts.
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
name = "12h_Donchian20_1dADX25_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ADX (14-period) on 1d timeframe
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d.shift(1))
    tr3 = np.abs(low_1d - close_1d.shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1).values
    
    # Directional Movement
    dm_plus = np.where((high_1d - high_1d.shift(1)) > (low_1d.shift(1) - low_1d), 
                       np.maximum(high_1d - high_1d.shift(1), 0), 0)
    dm_minus = np.where((low_1d.shift(1) - low_1d) > (high_1d - high_1d.shift(1)), 
                        np.maximum(low_1d.shift(1) - low_1d, 0), 0)
    
    # Smooth TR, DM+ and DM- (14-period)
    tr_ma = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_ma = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_ma = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_ma / tr_ma
    di_minus = 100 * dm_minus_ma / tr_ma
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above upper Donchian channel + strong trend + volume confirmation
            if close[i] > high_max[i] and strong_trend and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian channel + strong trend + volume confirmation
            elif close[i] < low_min[i] and strong_trend and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower Donchian channel OR trend weakens
            if close[i] < low_min[i] or adx_aligned[i] <= 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper Donchian channel OR trend weakens
            if close[i] > high_max[i] or adx_aligned[i] <= 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals