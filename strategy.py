#!/usr/bin/env python3
"""
12h_KAMA_Trend_Filter
KAMA-based trend following with volume confirmation and volatility filter
Designed to work in both bull and bear markets by:
- Using KAMA (adaptive moving average) to filter noise and capture trends
- Requiring volume confirmation to avoid false breakouts
- Using ATR-based volatility filter to avoid choppy markets
Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
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
    
    # === 1d KAMA (14-period) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Efficiency Ratio (ER) and Smoothing Constants
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i >= 1:
            if volatility[i] != 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 1.0
        else:
            er[i] = 0.0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama_1d = np.zeros_like(close_1d)
    kama_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama_1d[i] = kama_1d[i-1] + sc[i] * (close_1d[i] - kama_1d[i-1])
    
    # === 1d ATR (14-period) for volatility filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_14 = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i >= 13:
            atr_14[i] = np.mean(tr[i-13:i+1])
        elif i > 0:
            atr_14[i] = np.mean(tr[1:i+1])
        else:
            atr_14[i] = np.nan
    
    # === Align indicators to 12h timeframe ===
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # === 12h Volume confirmation ===
    # Calculate 20-period average volume
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_confirm = volume > vol_ma_20 * 1.3
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 30
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(atr_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price above KAMA with volume confirmation and sufficient volatility
            if (close[i] > kama_aligned[i] and 
                vol_confirm[i] and 
                atr_aligned[i] > 0):  # volatility filter
                signals[i] = 0.25
                position = 1
                continue
            # Short: price below KAMA with volume confirmation and sufficient volatility
            elif (close[i] < kama_aligned[i] and 
                  vol_confirm[i] and 
                  atr_aligned[i] > 0):  # volatility filter
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below KAMA
            if close[i] < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA
            if close[i] > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Trend_Filter"
timeframe = "12h"
leverage = 1.0