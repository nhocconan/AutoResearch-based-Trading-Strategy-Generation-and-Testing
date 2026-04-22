#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian(20) breakout with 12-hour volume spike and 1-day ATR volatility filter.
Long when price breaks above upper Donchian channel with volume > 1.5x 12h average volume and ATR(12h) > 0.5x ATR(1d).
Short when price breaks below lower Donchian channel with same volume and volatility filters.
Exit when price returns to the Donchian midpoint.
Designed for low trade frequency (~20-40/year) to avoid fee drag while capturing strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12-hour data for volume and volatility filters - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Load 1-day data for ATR volatility filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on 4h data
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max + low_min) / 2.0
    
    # Volume filter: 20-period average volume on 12h timeframe
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # ATR volatility filter: ATR(12h) > 0.5 * ATR(1d)
    # Calculate ATR for 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tr1_12h = np.maximum(high_12h[1:] - low_12h[1:], np.abs(high_12h[1:] - close_12h[:-1]))
    tr1_12h = np.maximum(tr1_12h, np.abs(low_12h[1:] - close_12h[:-1]))
    tr_12h = np.concatenate([[np.nan], tr1_12h])
    atr_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate ATR for 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1_1d = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr1_1d = np.maximum(tr1_1d, np.abs(low_1d[1:] - close_1d[:-1]))
    tr_1d = np.concatenate([[np.nan], tr1_1d])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(vol_ma_12h_aligned[i]) or np.isnan(atr_12h_aligned[i]) or np.isnan(atr_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian with volume and volatility confirmation
            if (close[i] > high_max[i] and 
                volume[i] > 1.5 * vol_ma_12h_aligned[i] and
                atr_12h_aligned[i] > 0.5 * atr_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian with volume and volatility confirmation
            elif (close[i] < low_min[i] and 
                  volume[i] > 1.5 * vol_ma_12h_aligned[i] and
                  atr_12h_aligned[i] > 0.5 * atr_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to Donchian midpoint
                if close[i] <= donchian_mid[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to Donchian midpoint
                if close[i] >= donchian_mid[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_Breakout_12hVol_1dATR_Filter"
timeframe = "4h"
leverage = 1.0