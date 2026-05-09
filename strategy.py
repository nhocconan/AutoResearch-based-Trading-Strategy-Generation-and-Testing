#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1w ADX trend filter and volume confirmation
# Long when price breaks above upper Donchian channel with weekly ADX > 25 and volume > 1.5x average
# Short when price breaks below lower Donchian channel with weekly ADX > 25 and volume > 1.5x average
# Exit when price retouches the opposite Donchian band or weekly ADX drops below 20
# Uses Donchian for breakout signals, ADX for trend strength filter, volume for conviction
# Designed to capture strong trend moves while avoiding choppy markets
# Target: 80-120 total trades over 4 years (20-30/year) with size 0.25

name = "4h_Donchian20_1wADX_Trend_Volume"
timeframe = "4h"
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
    
    # Calculate weekly Donchian channels (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-period high and low for Donchian channels
    high_20 = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 4h timeframe
    upper_dc = align_htf_to_ltf(prices, df_1w, high_20)
    lower_dc = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Calculate weekly ADX for trend filter (14-period)
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate True Range
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3 = abs(df_1w['low'] - df_1w['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    dm_plus = pd.Series(np.where((df_1w['high'] - df_1w['high'].shift(1)) > 
                                 (df_1w['low'].shift(1) - df_1w['low']), 
                                 np.maximum(df_1w['high'] - df_1w['high'].shift(1), 0), 0)).values
    dm_minus = pd.Series(np.where((df_1w['low'].shift(1) - df_1w['low']) > 
                                  (df_1w['high'] - df_1w['high'].shift(1)), 
                                  np.maximum(df_1w['low'].shift(1) - df_1w['low'], 0), 0)).values
    
    # Calculate smoothed TR and DM
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Calculate DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # Calculate DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_dc[i]) or np.isnan(lower_dc[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper Donchian, ADX > 25, volume spike
            if (close[i] > upper_dc[i] and 
                adx_aligned[i] > 25 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian, ADX > 25, volume spike
            elif (close[i] < lower_dc[i] and 
                  adx_aligned[i] > 25 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retouches lower Donchian or ADX drops below 20
            if (close[i] < lower_dc[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retouches upper Donchian or ADX drops below 20
            if (close[i] > upper_dc[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals