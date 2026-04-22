#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian(20) breakout with 1-day volume spike filter and ADX trend filter.
Long when price breaks above upper band, 1-day volume > 1.5x 20-day average, and ADX > 25.
Short when price breaks below lower band, 1-day volume > 1.5x 20-day average, and ADX > 25.
Exit when price crosses opposite Donchian band or ADX falls below 20.
Donchian provides clear breakout levels; volume confirms institutional interest; ADX filters ranging markets.
Designed for low trade frequency by requiring multiple confirmations and using 4h timeframe.
Works in both bull and bear markets by following breakouts with volume and trend confirmation.
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
    
    # Load 1-day data for volume filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d / vol_avg_1d  # Ratio of current volume to 20-day average
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Load 1-day data for ADX filter - ONCE before loop
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components
    plus_dm = np.zeros(len(high_1d))
    minus_dm = np.zeros(len(high_1d))
    tr = np.zeros(len(high_1d))
    
    for i in range(1, len(high_1d)):
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    
    # Smooth the values
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper band, volume spike, and ADX > 25
            if (close[i] > high_20[i] and 
                vol_spike_1d_aligned[i] > 1.5 and 
                adx_1d_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band, volume spike, and ADX > 25
            elif (close[i] < low_20[i] and 
                  vol_spike_1d_aligned[i] > 1.5 and 
                  adx_1d_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below lower band OR ADX falls below 20
                if (close[i] < low_20[i] or 
                    adx_1d_aligned[i] < 20):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above upper band OR ADX falls below 20
                if (close[i] > high_20[i] or 
                    adx_1d_aligned[i] < 20):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_Breakout_1dVolume_ADX"
timeframe = "4h"
leverage = 1.0