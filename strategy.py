#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volume confirmation and chop filter
# - Long when price breaks above Donchian(20) high + 1d volume > 1.5x 20-period average + chop > 61.8 (range)
# - Short when price breaks below Donchian(20) low + 1d volume > 1.5x 20-period average + chop > 61.8 (range)
# - Uses 12h timeframe for lower frequency trading to reduce fee drag
# - Volume confirmation ensures breakouts have conviction
# - Chop filter (range > 61.8) avoids whipsaws in strong trends
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volume and chop calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 12h Donchian channels
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Donchian(20) channels
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d average volume (20-period)
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Chopiness Index on 1d
    # ATR(14) calculation
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum.reduce([tr1, tr2, tr3])])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Max/min high-low over 14 periods
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(ATR14) / (max_high - min_low)) / log10(14)
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    range_14 = max_high_14 - min_low_14
    chop = 100 * np.log10(sum_atr_14 / range_14) / np.log10(14)
    
    # Align 1d indicators to 12h
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(avg_volume_1d_aligned[i]) or np.isnan(chop_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current 1d volume > 1.5x average
        volume_condition = volume_1d[i] > 1.5 * avg_volume_1d_aligned[i]
        
        # Chop condition: range-bound market (chop > 61.8)
        chop_condition = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long entry: break above Donchian high + volume + chop
            if close_12h[i] > donchian_high[i] and volume_condition and chop_condition:
                signals[i] = 0.25
                position = 1
            # Short entry: break below Donchian low + volume + chop
            elif close_12h[i] < donchian_low[i] and volume_condition and chop_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below Donchian low or conditions fail
            if close_12h[i] < donchian_low[i] or not (volume_condition and chop_condition):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above Donchian high or conditions fail
            if close_12h[i] > donchian_high[i] or not (volume_condition and chop_condition):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dVolumeChop"
timeframe = "12h"
leverage = 1.0