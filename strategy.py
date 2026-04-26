#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeRegime
Hypothesis: 4h Donchian(20) breakout with volume confirmation (>1.8x median) and choppiness regime filter (CHOP > 61.8) to capture mean-reversion in ranging markets. Enters long when price breaks above upper band with volume confirmation and choppy regime (fading extremes). Enters short when price breaks below lower band with volume confirmation and choppy regime. Exits on opposite Donchian breakout or when CHOP < 38.2 (trending regime). Uses discrete position sizing (0.25) to minimize churn. Target: 75-150 trades over 4 years. Works in both bull and bear markets by fading extremes in ranging conditions and avoiding strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.8x 50-period median
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=50, min_periods=50).median().values
    volume_confirm = volume > (1.8 * vol_median)
    
    # Choppiness regime filter (14-period)
    atr_14_list = []
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 * 14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20-period Donchian, 50-period volume median, 14-period chop)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(vol_median[i]) or np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: price breaks above upper Donchian + volume confirmation + choppy regime (CHOP > 61.8)
        if close[i] > highest_high_20[i] and volume_confirm[i] and chop[i] > 61.8:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below lower Donchian + volume confirmation + choppy regime (CHOP > 61.8)
        elif close[i] < lowest_low_20[i] and volume_confirm[i] and chop[i] > 61.8:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit conditions: opposite Donchian breakout OR trending regime (CHOP < 38.2)
        elif position == 1 and (close[i] < lowest_low_20[i] or chop[i] < 38.2):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > highest_high_20[i] or chop[i] < 38.2):
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

name = "4h_Donchian20_Breakout_VolumeRegime"
timeframe = "4h"
leverage = 1.0