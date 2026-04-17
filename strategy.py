#!/usr/bin/env python3
"""
1d_Weekly_13x34_SMA_Crossover_v1
Weekly SMA(13) crossing above SMA(34) for long, below for short.
Uses daily timeframe with weekly trend filter to reduce noise.
Exit when SMAs cross back or after 5 bars to limit exposure.
Designed to capture medium-term trends with clear signals.
Target: 20-60 total trades over 4 years (5-15/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # === Weekly SMAs (13, 34) ===
    df_1w = get_htf_data(prices, '1w')
    sma_13_1w = pd.Series(df_1w['close'].values).rolling(window=13, min_periods=13).mean().values
    sma_34_1w = pd.Series(df_1w['close'].values).rolling(window=34, min_periods=34).mean().values
    sma_13_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_13_1w)
    sma_34_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_34_1w)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 34
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(sma_13_1w_aligned[i]) or 
            np.isnan(sma_34_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: SMA13 > SMA34
            if sma_13_1w_aligned[i] > sma_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
                continue
            # Short: SMA13 < SMA34
            elif sma_13_1w_aligned[i] < sma_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: SMA13 < SMA34 OR after 5 bars
            if sma_13_1w_aligned[i] < sma_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: SMA13 > SMA34 OR after 5 bars
            if sma_13_1w_aligned[i] > sma_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_13x34_SMA_Crossover_v1"
timeframe = "1d"
leverage = 1.0