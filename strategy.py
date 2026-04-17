#!/usr/bin/env python3
"""
4h_Stochastic_Breakout_Trend_v1
Stochastic breakout with trend filter and volume confirmation.
Long when %K crosses above 80 in uptrend, short when %K crosses below 20 in downtrend.
Exit when %K crosses 50 level.
Uses 1d ADX for trend strength and 1d volume spike for confirmation.
Designed to capture momentum bursts with filtered entries.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Stochastic Oscillator (14,3,3) ===
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values
    
    # === Volume spike (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma + 1e-10)
    
    # === 1d ADX for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate DM
    plus_dm_1d = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                          np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm_1d = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                           np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm_1d = np.concatenate([[0], plus_dm_1d])
    minus_dm_1d = np.concatenate([[0], minus_dm_1d])
    
    plus_di_1d = 100 * pd.Series(plus_dm_1d).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_1d + 1e-10)
    minus_di_1d = 100 * pd.Series(minus_dm_1d).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_1d + 1e-10)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 4h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 1d Volume average for spike detection ===
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 30
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(k_percent[i]) or 
            np.isnan(d_percent[i]) or 
            np.isnan(volume_ratio[i]) or 
            np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 1d average
        volume_spike = volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Stochastic crosses above 80, strong uptrend, volume spike
            if (k_percent[i-1] <= 80 and k_percent[i] > 80 and 
                adx_1d_aligned[i] > 25 and 
                volume_spike):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Stochastic crosses below 20, strong downtrend, volume spike
            elif (k_percent[i-1] >= 20 and k_percent[i] < 20 and 
                  adx_1d_aligned[i] > 25 and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: Stochastic crosses 50 level
        elif position == 1:
            # Exit long: %K crosses below 50
            if k_percent[i-1] >= 50 and k_percent[i] < 50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: %K crosses above 50
            if k_percent[i-1] <= 50 and k_percent[i] > 50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Stochastic_Breakout_Trend_v1"
timeframe = "4h"
leverage = 1.0