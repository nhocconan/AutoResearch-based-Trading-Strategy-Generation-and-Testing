#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + choppiness regime filter
# Long when: price breaks above Donchian(20) high + 1d volume > 2.0x 20-period MA + CHOP(14) > 61.8 (range)
# Short when: price breaks below Donchian(20) low + 1d volume > 2.0x 20-period MA + CHOP(14) > 61.8 (range)
# Exit when: price re-enters Donchian channel OR CHOP(14) < 38.2 (trending)
# Uses Donchian for structure, volume for conviction, chop for ranging markets where mean reversion works
# Timeframe: 4h, HTF: 1d. Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

name = "4h_Donchian20_1dVolumeSpike_ChopRange"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels on 4h
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    # Get 1d data ONCE before loop for volume and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume spike filter: volume > 2.0x 20-period MA
    if len(volume_1d) >= 20:
        vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
        volume_spike_1d = volume_1d > (2.0 * vol_ma_20_1d)
    else:
        volume_spike_1d = np.full(len(volume_1d), False)
    
    # Calculate 1d choppiness index: CHOP(14)
    # CHOP = 100 * log10(sum(ATR(1),14) / (log10(highest_high - lowest_low) * 14)) / log10(14)
    if len(high_1d) >= 14:
        # True Range
        tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
        tr1 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
        tr = np.concatenate([[np.nan], tr1])  # align with index
        
        # ATR(14)
        atr_1 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
        
        # Sum of ATR over 14 periods
        sum_atr_14 = pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values
        
        # Highest high and lowest low over 14 periods
        highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        
        # Avoid division by zero
        range_14 = highest_high_14 - lowest_low_14
        range_14 = np.where(range_14 == 0, np.nan, range_14)
        
        # Chop calculation
        chop = 100 * np.log10(sum_atr_14 / (np.log10(range_14) * 14)) / np.log10(14)
        chop = np.where(np.isnan(chop) | np.isinf(chop), 50.0, chop)  # default to neutral
    else:
        chop = np.full(len(high_1d), 50.0)
    
    # Align 1d indicators to 4h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Donchian breakout up + volume spike + chop > 61.8 (range)
            if (close[i] > donchian_high[i] and 
                volume_spike_aligned[i] and 
                chop_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Short conditions: Donchian breakout down + volume spike + chop > 61.8 (range)
            elif (close[i] < donchian_low[i] and 
                  volume_spike_aligned[i] and 
                  chop_aligned[i] > 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Donchian channel OR chop < 38.2 (trending)
            if (close[i] <= donchian_high[i] or chop_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Donchian channel OR chop < 38.2 (trending)
            if (close[i] >= donchian_low[i] or chop_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals