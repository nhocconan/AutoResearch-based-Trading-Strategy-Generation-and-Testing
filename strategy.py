#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above 20-period high with ADX>25 and volume > 1.5x average
# Short when price breaks below 20-period low with ADX>25 and volume > 1.5x average
# Exit when price reverses to opposite Donchian boundary or ADX < 20 (trend weak)
# Uses Donchian for breakout structure, ADX for trend strength, volume for conviction
# Designed to capture sustained moves in both bull and bear markets with low frequency
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25

name = "12h_Donchian_Breakout_1dADX_Volume"
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
    
    # Calculate 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate True Range and directional movement for ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_period = 14
    atr = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False).mean()
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/tr_period, adjust=False).mean()
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/tr_period, adjust=False).mean()
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False).mean()
    adx_values = adx.values
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Calculate 12h Donchian channels (20-period)
    donchian_window = 20
    upper_channel = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max()
    lower_channel = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min()
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, donchian_window)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(upper_channel.iloc[i]) or 
            np.isnan(lower_channel.iloc[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper channel, ADX>25, volume spike
            if (close[i] > upper_channel.iloc[i] and 
                adx_aligned[i] > 25 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower channel, ADX>25, volume spike
            elif (close[i] < lower_channel.iloc[i] and 
                  adx_aligned[i] > 25 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to lower channel or ADX < 20 (trend weak)
            if (close[i] < lower_channel.iloc[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to upper channel or ADX < 20 (trend weak)
            if (close[i] > upper_channel.iloc[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals