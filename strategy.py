#!/usr/bin/env python3
"""
4h Choppiness Index + Donchian Breakout + Volume Spike
Long: Donchian high breakout + volume > 2x 4h volume SMA(20) + CHOP > 61.8 (range)
Short: Donchian low breakout + volume > 2x 4h volume SMA(20) + CHOP > 61.8 (range)
Exit: Opposite Donchian breakout or CHOP < 38.2 (trend)
Uses Choppiness Index to identify ranging markets where Donchian breakouts are more reliable.
Designed to work in both bull and bear markets by focusing on range-bound conditions.
Target: 50-150 total trades over 4 years (12-37/year)
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
    
    # Calculate 14-period True Range for Choppiness Index
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[high[0] - low[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate ATR(14) for Choppiness Index
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing
    
    # Calculate Choppiness Index: 100 * log10(sum(ATR,14) / (max(high,14) - min(low,14))) / log10(14)
    chop = np.full(n, np.nan)
    for i in range(13, n):
        atr_sum = np.sum(atr[i-13:i+1])
        max_high = np.max(high[i-13:i+1])
        min_low = np.min(low[i-13:i+1])
        if max_high != min_low:
            chop[i] = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    
    # Donchian Channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(19, n):
        donch_high[i] = np.max(high[i-19:i+1])
        donch_low[i] = np.min(low[i-19:i+1])
    
    # Volume SMA(20)
    vol_sma = np.full(n, np.nan)
    for i in range(19, n):
        vol_sma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 20  # need Donchian and volume data
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(vol_sma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma[i]
        upper = donch_high[i]
        lower = donch_low[i]
        chop_val = chop[i]
        
        if position == 0:
            # Long: Donchian high breakout + volume spike + ranging market (CHOP > 61.8)
            if price > upper and vol > 2.0 * vol_sma_val and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: Donchian low breakout + volume spike + ranging market (CHOP > 61.8)
            elif price < lower and vol > 2.0 * vol_sma_val and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Opposite breakout or trending market (CHOP < 38.2)
            if price < lower or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Opposite breakout or trending market (CHOP < 38.2)
            if price > upper or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Choppiness_Donchian_Breakout_Volume"
timeframe = "4h"
leverage = 1.0