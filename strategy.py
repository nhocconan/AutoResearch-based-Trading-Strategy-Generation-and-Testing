#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume spike and chop regime filter.
# Long when price breaks above 4h Donchian upper channel (20-period high) AND 
# volume > 1.8x 20-period average AND chop > 61.8 (ranging market for mean reversion).
# Short when price breaks below 4h Donchian lower channel (20-period low) AND 
# volume > 1.8x 20-period average AND chop > 61.8.
# Exit when price crosses back inside the Donchian channel.
# Chop filter ensures we trade mean reversion in ranging markets, avoiding trending whipsaws.
# Target: 25-40 trades/year (100-160 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by capturing mean reversion in ranges.

name = "4h_Donchian_Breakout_Volume_Chop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma20)
    
    # Chop index (14-period) for regime detection
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10((atr14 * 14) / (highest_high14 - lowest_low14)) / np.log10(14)
    chop[np.isnan(chop) | (highest_high14 == lowest_low14)] = 50  # Default when range is 0
    
    chop_filter = chop > 61.8  # Ranging market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(chop_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper, volume filter, chop filter
            long_cond = (close[i] > high_roll[i-1]) and volume_filter[i] and chop_filter[i]
            # Short conditions: price breaks below Donchian lower, volume filter, chop filter
            short_cond = (close[i] < low_roll[i-1]) and volume_filter[i] and chop_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian lower
            if close[i] < low_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Donchian upper
            if close[i] > high_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals