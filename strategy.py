#!/usr/bin/env python3
"""
4x4 Grid Strategy: 4h Donchian Breakout with 4h Volume and 4h ADX Filter
Hypothesis: Donchian channel breakouts filtered by volume expansion and ADX trend strength
work in both bull and bear markets by capturing momentum bursts while avoiding whipsaws.
Designed for 15-25 trades/year per symbol with clear entry/exit rules.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4x4_grid_strategy"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) on 4h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    # ADX (14-period) for trend strength
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(low, 1)), np.abs(low - np.roll(high, 1))))
    tr[0] = high[0] - low[0]
    
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # ADX filter: only trade when trend is strong (ADX > 25)
    strong_trend = adx > 25
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or
            np.isnan(vol_spike[i]) or
            np.isnan(strong_trend[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low
            if close[i] < low_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high
            if close[i] > high_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high with volume spike and strong trend
            if (high[i] > high_20[i-1] and 
                vol_spike[i] and 
                strong_trend[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low with volume spike and strong trend
            elif (low[i] < low_20[i-1] and 
                  vol_spike[i] and 
                  strong_trend[i]):
                position = -1
                signals[i] = -0.25
    
    return signals