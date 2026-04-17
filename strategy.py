#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_Volume_Confirmation_v1
KAMA(10) direction for trend, volume > 1.5x MA(20) for confirmation.
Exit when KAMA flips direction or volume drops below average.
Designed to work in both bull and bear markets by following adaptive trend.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === KAMA(10) ===
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, 10))
    change[0:10] = 0  # First 10 values invalid
    
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # Temporary fix, will compute properly below
    # Recompute volatility properly
    volatility = np.zeros(n)
    for i in range(1, n):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    # Volatility over 10 periods
    vol_10 = np.zeros(n)
    for i in range(10, n):
        vol_10[i] = volatility[i] - volatility[i-10]
    
    # Avoid division by zero
    er = np.zeros(n)
    mask = vol_10 != 0
    er[mask] = change[mask] / vol_10[mask]
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === Volume confirmation ===
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 30
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: KAMA rising AND volume > 1.5x average
            if (kama[i] > kama[i-1] and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
                continue
            # Short: KAMA falling AND volume > 1.5x average
            elif (kama[i] < kama[i-1] and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: KAMA falling OR volume < average
            if (kama[i] < kama[i-1] or 
                vol_ratio[i] < 1.0):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA rising OR volume < average
            if (kama[i] > kama[i-1] or 
                vol_ratio[i] < 1.0):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Trend_With_Volume_Confirmation_v1"
timeframe = "4h"
leverage = 1.0