#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d ADX(14) trend filter and volume confirmation
# Long when price breaks above 20-period high with ADX > 25 and volume > 1.5x average
# Short when price breaks below 20-period low with ADX > 25 and volume > 1.5x average
# Exit when price reverses to opposite Donchian level or ADX falls below 20
# Uses Donchian channels for breakout signals, ADX for trend strength, volume for conviction
# Designed to capture strong trends in both bull and bear markets with controlled frequency
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25

name = "12h_Donchian20_1dADX25_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

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
    
    # Calculate 12h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # Calculate 1d ADX(14) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    dm_plus = pd.Series(index=df_1d.index, dtype=float)
    dm_minus = pd.Series(index=df_1d.index, dtype=float)
    dm_plus[df_1d['high'] - df_1d['high'].shift(1) > df_1d['low'].shift(1) - df_1d['low']] = \
        np.where(df_1d['high'] - df_1d['high'].shift(1) > df_1d['low'].shift(1) - df_1d['low'], 
                 df_1d['high'] - df_1d['high'].shift(1), 0)
    dm_minus[df_1d['low'].shift(1) - df_1d['low'] > df_1d['high'] - df_1d['high'].shift(1)] = \
        np.where(df_1d['low'].shift(1) - df_1d['low'] > df_1d['high'] - df_1d['high'].shift(1), 
                 df_1d['low'].shift(1) - df_1d['low'], 0)
    
    # Smooth the values
    atr = tr.rolling(window=14, min_periods=14).mean()
    dm_plus_smooth = dm_plus.rolling(window=14, min_periods=14).mean()
    dm_minus_smooth = dm_minus.rolling(window=14, min_periods=14).mean()
    
    # Calculate DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # Calculate DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    # Align indicators to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20.values)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20.values)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicator calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above 20-period high, ADX > 25, volume spike
            if (close[i] > high_20_aligned[i] and 
                adx_aligned[i] > 25 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 20-period low, ADX > 25, volume spike
            elif (close[i] < low_20_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reverses to 20-period low or ADX falls below 20
            if (close[i] < low_20_aligned[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reverses to 20-period high or ADX falls below 20
            if (close[i] > high_20_aligned[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals