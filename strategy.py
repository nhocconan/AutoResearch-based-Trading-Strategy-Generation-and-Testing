#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d ADX trend filter + volume confirmation
# Donchian breakout provides clear structure-based entries in both bull/bear markets
# 1d ADX > 25 filters for trending markets, avoids chop
# Volume confirmation ensures conviction behind breakouts
# Exits on opposite Donchian breakout or ADX < 20 (trend weakening)
# Timeframe: 12h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Donchian20_1dADX_VolumeConfirm"
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
    
    # Get 1d data ONCE before loop for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    if len(close_1d) >= 14:
        # True Range
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                           np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
        dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                            np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed values
        atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / np.where(atr_14 == 0, np.nan, atr_14)
        di_minus = 100 * dm_minus_smooth / np.where(atr_14 == 0, np.nan, atr_14)
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, np.nan, (di_plus + di_minus))
        adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    else:
        adx_1d = np.full(len(close_1d), np.nan)
    
    # Align 1d ADX to 12h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 12h Donchian channels (20-period)
    if len(high) >= 20:
        # Rolling max/min for Donchian channels
        high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
        low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_upper = high_max
        donchian_lower = low_min
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
    
    # Volume confirmation on 12h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.5 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: ADX>25 AND price breaks above Donchian upper AND volume spike
            if (adx_1d_aligned[i] > 25 and 
                close[i] > donchian_upper[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: ADX>25 AND price breaks below Donchian lower AND volume spike
            elif (adx_1d_aligned[i] > 25 and 
                  close[i] < donchian_lower[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian lower OR ADX < 20 (trend weakening)
            if close[i] < donchian_lower[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian upper OR ADX < 20 (trend weakening)
            if close[i] > donchian_upper[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals