#!/usr/bin/env python3
"""
4h_KAMA_Trend_with_1dKAMA_Confirmation
Hypothesis: KAMA adapts to market noise, providing reliable trend direction. 
Using 1d KAMA as higher-timeframe filter ensures we only trade in the direction of the daily trend, 
reducing false signals in choppy markets. KAMA crossover on 4h provides entry, with volume confirmation.
Designed to work in both bull and bear markets by following the dominant trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA on 4h
    def kama(close, er_period=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close)).cumsum()
        volatility = np.diff(np.concatenate([[0], volatility]))
        er = np.zeros_like(close)
        for i in range(len(close)):
            if volatility[i] != 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_4h = kama(close, er_period=10, fast_sc=2, slow_sc=30)
    
    # Get 1d data for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    kama_1d = kama(close_1d, er_period=10, fast_sc=2, slow_sc=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(kama_4h[i]) or 
            np.isnan(kama_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_fast = kama_4h[i]
        kama_1d_val = kama_1d_aligned[i]
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: 4h KAMA price crossover up AND 1d KAMA uptrend AND volume confirmation
            if price > kama_fast and kama_4h[i] > kama_4h[i-1] and kama_1d[i] > kama_1d[i-1] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: 4h KAMA price crossover down AND 1d KAMA downtrend AND volume confirmation
            elif price < kama_fast and kama_4h[i] < kama_4h[i-1] and kama_1d[i] < kama_1d[i-1] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses below KAMA OR 1d trend changes
            if price < kama_fast or kama_1d[i] < kama_1d[i-1]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses above KAMA OR 1d trend changes
            if price > kama_fast or kama_1d[i] > kama_1d[i-1]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Trend_with_1dKAMA_Confirmation"
timeframe = "4h"
leverage = 1.0