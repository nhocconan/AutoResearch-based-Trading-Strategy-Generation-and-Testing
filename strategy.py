#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_1d_ADX_Filter_v2
Hypothesis: Use 1d ADX > 25 as trend filter with KAMA direction on 4h for entries. Long when KAMA rising and price > KAMA, short when KAMA falling and price < KAMA. Exit when price crosses KAMA in opposite direction. Designed for fewer trades (target 20-40/year) with trend alignment to work in both bull and bear markets.
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
    
    # Calculate ADX on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up = df_1d['high'] - df_1d['high'].shift(1)
    down = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di14 = 100 * plus_dm14 / tr14
    minus_di14 = 100 * minus_dm14 / tr14
    
    # DX and ADX
    dx = 100 * abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 4h timeframe (wait for previous day's close)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # KAMA on 4h
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = abs(close_s - close_s.shift(10))
    volatility = abs(close_s - close_s.shift(1)).rolling(window=10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    # KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(adx_aligned[i]) or np.isnan(kama[i]):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        kama_val = kama[i]
        
        if position == 0:
            # Long: ADX > 25 (trending) AND KAMA rising AND price > KAMA
            if adx_val > 25 and kama[i] > kama[i-1] and close[i] > kama_val:
                signals[i] = size
                position = 1
            # Short: ADX > 25 (trending) AND KAMA falling AND price < KAMA
            elif adx_val > 25 and kama[i] < kama[i-1] and close[i] < kama_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA
            if close[i] < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above KAMA
            if close[i] > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_KAMA_Trend_With_1d_ADX_Filter"
timeframe = "4h"
leverage = 1.0