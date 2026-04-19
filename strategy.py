#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout (20-period) + volume spike + Choppiness regime filter.
# Donchian provides clear breakout signals in trends; volume confirms strength; Choppiness avoids whipsaws in ranges.
# Works in both bull/bear: breakouts capture momentum, Choppiness filter reduces false signals in consolidation.
# Target: 20-40 trades/year to minimize fee drag. Size: 0.25.

name = "4h_Donchian_Volume_Chop"
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
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Choppiness Index (14-period)
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr_14 / (max_hh - min_ll)) / np.log10(14)
    chop = np.where((max_hh - min_ll) == 0, 50, chop)  # avoid div/0
    
    # Volume spike: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(chop[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper Donchian with volume and trending market (Chop < 61.8)
            if (close[i] > high_20[i] and 
                volume_spike[i] and 
                chop[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume and trending market (Chop < 61.8)
            elif (close[i] < low_20[i] and 
                  volume_spike[i] and 
                  chop[i] < 61.8):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price touches lower Donchian or chop becomes high (range)
            if (close[i] < low_20[i]) or (chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price touches upper Donchian or chop becomes high (range)
            if (close[i] > high_20[i]) or (chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals