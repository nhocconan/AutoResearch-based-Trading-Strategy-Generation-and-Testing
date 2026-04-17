#!/usr/bin/env python3
"""
12h_Donchian_20_Breakout_With_Volume_Filter
Long when price breaks above Donchian high(20) with volume > 1.5x median volume(20).
Short when price breaks below Donchian low(20) with volume > 1.5x median volume(20).
Exit when price re-enters the Donchian channel.
Uses 1d ADX > 20 as trend filter to avoid whipsaws in ranging markets.
Designed for 12h timeframe to capture medium-term trends with volume confirmation.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # === Donchian Channel (20) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume filter: volume > 1.5x median volume(20) ===
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    vol_filter = volume > (1.5 * vol_median)
    
    # === 1d ADX(14) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    
    # Directional Movement
    plus_dm_1d = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                          np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm_1d = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                           np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm_1d = np.concatenate([[0], plus_dm_1d])
    minus_dm_1d = np.concatenate([[0], minus_dm_1d])
    
    # ADX calculation
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm_1d).rolling(window=14, min_periods=14).sum().values / (atr_1d * 14)
    minus_di_1d = 100 * pd.Series(minus_dm_1d).rolling(window=14, min_periods=14).sum().values / (atr_1d * 14)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(vol_filter[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above Donchian high, volume filter, ADX > 20
            if (close[i] > donch_high[i] and 
                vol_filter[i] and 
                adx_1d_aligned[i] > 20):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below Donchian low, volume filter, ADX > 20
            elif (close[i] < donch_low[i] and 
                  vol_filter[i] and 
                  adx_1d_aligned[i] > 20):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price re-enters Donchian channel (below Donchian high)
            if close[i] < donch_high[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price re-enters Donchian channel (above Donchian low)
            if close[i] > donch_low[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_20_Breakout_With_Volume_Filter"
timeframe = "12h"
leverage = 1.0