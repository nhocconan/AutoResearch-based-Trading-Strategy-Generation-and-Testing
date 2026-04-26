#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeChopFilter
Hypothesis: 4h Donchian(20) breakout with volume spike and choppiness regime filter.
Enters long on upper band breakout when volume > 2x EMA20 volume and chop > 61.8 (range).
Enters short on lower band breakout when volume > 2x EMA20 volume and chop > 61.8 (range).
Exits on opposite band touch.
Designed for 75-200 total trades over 4 years (19-50/year) to avoid fee drag.
Uses discrete position sizing (0.25) to minimize churn. Works in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) on 4h timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    # Choppiness Index (14-period) - uses true range and ATR
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.concatenate([[close[0]], close[:-1]])))
    tr3 = pd.Series(np.abs(low - np.concatenate([[close[0]], close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    sum_tr = tr.rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)  # replace NaN with neutral value
    
    # Regime filter: chop > 61.8 indicates ranging market (good for mean reversion/breakout fade)
    chop_range = chop > 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20-period Donchian and 14-period chop)
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(chop_range[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above upper Donchian + volume spike + choppy regime
        if close[i] > highest_high[i] and volume_spike[i] and chop_range[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: break below lower Donchian + volume spike + choppy regime
        elif close[i] < lowest_low[i] and volume_spike[i] and chop_range[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price touches opposite Donchian band
        elif position == 1 and close[i] < lowest_low[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > highest_high[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Donchian20_Breakout_VolumeChopFilter"
timeframe = "4h"
leverage = 1.0