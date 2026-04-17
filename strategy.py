#!/usr/bin/env python3
"""
4h Donchian Breakout + Volume Spike + Choppiness Filter (Regime)
Long: Price breaks above Donchian(20) high + volume > 1.5x volume SMA(20) + choppiness < 61.8 (trending)
Short: Price breaks below Donchian(20) low + volume > 1.5x volume SMA(20) + choppiness < 61.8 (trending)
Exit: Opposite Donchian breakout or choppiness > 61.8 (range)
Uses choppiness to filter out ranging markets, focusing on trending periods where breakouts work best.
Designed to work in both bull and bear markets by using volume confirmation and regime filter.
Target: 75-200 total trades over 4 years (19-50/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume SMA(20)
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate choppiness index (14-period)
    atr = pd.Series(np.sqrt((high - low)**2)).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr.sum() / (highest_high - lowest_low)) / np.log10(14)
    # Fix: rolling sum of ATR
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    hh_ll = highest_high - lowest_low
    chop = 100 * np.log10(atr_sum / hh_ll) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 20  # need Donchian and choppiness
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(vol_sma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma[i]
        chop_val = chop[i]
        
        if position == 0:
            # Long: break above Donchian high + volume spike + trending (chop < 61.8)
            if price > donch_high[i] and vol > 1.5 * vol_sma_val and chop_val < 61.8:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + volume spike + trending (chop < 61.8)
            elif price < donch_low[i] and vol > 1.5 * vol_sma_val and chop_val < 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below Donchian low OR chop > 61.8 (range)
            if price < donch_low[i] or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above Donchian high OR chop > 61.8 (range)
            if price > donch_high[i] or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0