#!/usr/bin/env python3
"""
6h_supertrend_1w_trend_volume_v1
Hypothesis: Supertrend on 1w defines major trend (bull/bear), 6h Supertrend provides entry timing with volume confirmation.
In bull markets (1w Supertrend up), go long when 6h Supertrend flips up with volume; in bear markets (1w Supertrend down), go short when 6h Supertrend flips down with volume.
Uses ATR-based trailing stops via Supertrend logic. Targets 15-30 trades/year by requiring weekly trend alignment and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_supertrend_1w_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator. Returns (supertrend, direction) where direction: 1=uptrend, -1=downtrend"""
    hl2 = (high + low) / 2
    atr = pd.Series(high - low).rolling(window=period, min_periods=period).mean()
    atr = pd.Series(np.maximum(high - low, np.maximum(abs(high - pd.Series(close).shift(1)), abs(low - pd.Series(close).shift(1))))).rolling(window=period, min_periods=period).mean()
    
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr
    
    supertrend = np.full_like(close, np.nan, dtype=float)
    direction = np.full_like(close, 1, dtype=float)  # start with uptrend
    
    for i in range(period, len(close)):
        if close[i-1] > supertrend[i-1]:
            supertrend[i] = max(lowerband[i], supertrend[i-1])
        else:
            supertrend[i] = min(upperband[i], supertrend[i-1])
        
        if close[i] > upperband[i-1]:
            direction[i] = 1
        elif close[i] < lowerband[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            
        if direction[i] == 1 and supertrend[i] < supertrend[i-1]:
            supertrend[i] = upperband[i]
        if direction[i] == -1 and supertrend[i] > supertrend[i-1]:
            supertrend[i] = lowerband[i]
            
    return supertrend, direction

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w Supertrend for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    st_1w, dir_1w = supertrend(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, period=10, multiplier=3.0)
    st_1w_6h = align_htf_to_ltf(prices, df_1w, st_1w)
    dir_1w_6h = align_htf_to_ltf(prices, df_1w, dir_1w)
    
    # 6h Supertrend for entry timing
    st_6h, dir_6h = supertrend(high, low, close, period=10, multiplier=3.0)
    
    # 20-period SMA for volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(st_1w_6h[i]) or np.isnan(dir_1w_6h[i]) or 
            np.isnan(st_6h[i]) or np.isnan(dir_6h[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: 6h Supertrend flips down OR weekly trend turns down
            if dir_6h[i] == -1 or dir_1w_6h[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: 6h Supertrend flips up OR weekly trend turns up
            if dir_6h[i] == 1 or dir_1w_6h[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: 6h Supertrend flips up + volume + weekly uptrend
            if (dir_6h[i] == 1 and dir_6h[i-1] == -1 and  # fresh flip up
                vol_confirm and 
                dir_1w_6h[i] == 1):
                position = 1
                signals[i] = 0.25
            # Short: 6h Supertrend flips down + volume + weekly downtrend
            elif (dir_6h[i] == -1 and dir_6h[i-1] == 1 and  # fresh flip down
                  vol_confirm and 
                  dir_1w_6h[i] == -1):
                position = -1
                signals[i] = -0.25
    
    return signals