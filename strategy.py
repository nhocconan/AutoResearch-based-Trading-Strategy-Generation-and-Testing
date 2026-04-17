#!/usr/bin/env python3
"""
4h Donchian Breakout + Volume Spike + Chop Filter
Long: Price breaks above Donchian(20) high + volume > 2x volume SMA(20) + CHOP > 61.8
Short: Price breaks below Donchian(20) low + volume > 2x volume SMA(20) + CHOP > 61.8
Exit: Opposite Donchian break or CHOP < 38.2
Donchian provides trend structure, volume confirms breakout strength, chop filter avoids whipsaws in ranging markets.
Target: 150-250 total trades over 4 years (38-63/year) - within safe limits for 4h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14-period) for regime filter
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Chop: 100 * log15(sum(ATR)/ (max(high)-min(low)))
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log15((atr * 14) / (highest_high - lowest_low + 1e-10))
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)  # avoid div by zero
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 20  # need Donchian and chop
    
    for i in range(start_idx, n):
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(vol_sma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma[i]
        chop_val = chop[i]
        
        if position == 0:
            # Long: Donchian breakout up + volume spike + choppy market (mean reversion setup)
            if price > high_max[i-1] and vol > 2.0 * vol_sma_val and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + volume spike + choppy market
            elif price < low_min[i-1] and vol > 2.0 * vol_sma_val and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Donchian breakdown or trending market (chop < 38.2)
            if price < low_min[i-1] or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Donchian breakout up or trending market
            if price > high_max[i-1] or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_Chop"
timeframe = "4h"
leverage = 1.0