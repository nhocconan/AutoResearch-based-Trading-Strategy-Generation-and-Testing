#!/usr/bin/env python3
"""
4h_PriceAction_CloseAboveYesterdayHigh_1dEMA50
Hypothesis: Enters long when 4h close exceeds yesterday's high with price above 1d EMA50 (uptrend).
Enters short when 4h close falls below yesterday's low with price below 1d EMA50 (downtrend).
Uses pure price action with trend filter for robustness in both bull and bear markets.
Designed for low trade frequency (20-40 trades/year) to minimize fee drag.
"""

name = "4h_PriceAction_CloseAboveYesterdayHigh_1dEMA50"
timeframe = "4h"
leverage = 1.0

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
    
    # === 1D Data for Trend Filter and Yesterday's Levels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA50 for trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Previous day's high and low (yesterday's levels)
    ph_1d = high_1d  # previous day high
    pl_1d = low_1d   # previous day low
    
    # Align yesterday's levels to 4h
    ph_aligned = align_htf_to_ltf(prices, df_1d, ph_1d)
    pl_aligned = align_htf_to_ltf(prices, df_1d, pl_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers 1d EMA50)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ph_aligned[i]) or np.isnan(pl_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: close above yesterday's high with uptrend
            if (close[i] > ph_aligned[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: close below yesterday's low with downtrend
            elif (close[i] < pl_aligned[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below yesterday's low (trend invalidation)
            if close[i] < pl_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: close above yesterday's high (trend invalidation)
            if close[i] > ph_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals