#!/usr/bin/env python3
"""
12h_1d_trix_volume_crossover
Trix (12) crossover on 1d timeframe with volume confirmation on 12h.
Long when TRIX crosses above zero with volume > 1.5x average, short when crosses below zero.
Exit on opposite TRIX cross.
Uses TRIX as a momentum oscillator that filters noise and works in both trending and ranging markets.
Target: 15-30 trades/year on 12h timeframe for low frequency and minimal fee drag.
"""

name = "12h_1d_trix_volume_crossover"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for TRIX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # TRIX calculation: triple EMA of percentage change
    # Step 1: Calculate % change
    pct_change = np.diff(close_1d) / close_1d[:-1]
    pct_change = np.insert(pct_change, 0, 0)  # First value is 0
    
    # Step 2: Triple EMA
    ema1 = pd.Series(pct_change).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = ema3 * 100  # Scale for readability
    
    # Align TRIX to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Volume confirmation on 12h: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(trix_aligned[i]) or np.isnan(trix_aligned[i-1]):
            signals[i] = 0.0
            continue
        
        # Long entry: TRIX crosses above zero with volume confirmation
        if trix_aligned[i-1] <= 0 and trix_aligned[i] > 0 and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short entry: TRIX crosses below zero with volume confirmation
        elif trix_aligned[i-1] >= 0 and trix_aligned[i] < 0 and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit on opposite TRIX cross
        elif position == 1 and trix_aligned[i-1] >= 0 and trix_aligned[i] < 0:
            position = 0
            signals[i] = 0.0
        elif position == -1 and trix_aligned[i-1] <= 0 and trix_aligned[i] > 0:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals