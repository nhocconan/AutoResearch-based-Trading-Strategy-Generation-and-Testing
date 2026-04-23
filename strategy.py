#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d ADX(14) trend filter and volume confirmation.
Long when price breaks above upper Donchian channel (20-period high) AND 1d ADX > 25 (strong trend) AND volume > 1.5x average.
Short when price breaks below lower Donchian channel (20-period low) AND 1d ADX > 25 (strong trend) AND volume > 1.5x average.
Exit when price reverts to the 20-period midpoint (mean reversion) OR ADX drops below 20 (weakening trend).
Uses 6h timeframe for moderate trade frequency with tight entry conditions to avoid fee drag.
1d ADX provides regime filter to trade only in strong trending markets, reducing whipsaws.
Target: 75-150 trades over 4 years (19-38/year).
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
    
    # Load 6h data for Donchian channel calculation - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate Donchian channel (20-period) on 6h
    # Upper = rolling max(high, 20), Lower = rolling min(low, 20), Middle = (Upper + Lower) / 2
    high_roll_max = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll_max
    donchian_lower = low_roll_min
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Load 1d data for ADX(14) trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period) on 1d
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # Directional Indicators
    plus_di_14 = 100 * plus_dm_14 / tr_14
    minus_di_14 = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx_denom = plus_di_14 + minus_di_14
    dx_denom = np.where(dx_denom == 0, 1e-10, dx_denom)  # Avoid division by zero
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / dx_denom
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 6h indicators to 6h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_6h, donchian_middle)
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_val = donchian_upper_aligned[i]
        lower_val = donchian_lower_aligned[i]
        middle_val = donchian_middle_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian AND ADX > 25 (strong trend) AND volume spike
            if (price > upper_val and adx_val > 25.0 and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND ADX > 25 (strong trend) AND volume spike
            elif (price < lower_val and adx_val > 25.0 and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to middle OR ADX drops below 20 (weakening trend)
                if price <= middle_val or adx_val < 20.0:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to middle OR ADX drops below 20 (weakening trend)
                if price >= middle_val or adx_val < 20.0:
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