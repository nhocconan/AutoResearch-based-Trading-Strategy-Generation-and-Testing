#!/usr/bin/env python3
"""
12h_1d_TRIX_Signal_Volume_Confirmation_v1
Hypothesis: TRIX momentum on 12h timeframe with volume confirmation on 12h.
Long when TRIX crosses above zero with volume above average.
Short when TRIX crosses below zero with volume above average.
Exit when TRIX returns to zero.
Uses daily trend filter: only take long when price above 200-day EMA, short when below.
Designed for low trade frequency (<30/year) to minimize fee drag.
Works in bull/bear by following TRIX momentum with daily trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter (200-day EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Load 12h data for TRIX and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate TRIX (15-period EMA of EMA of EMA of close)
    ema1 = pd.Series(close_12h).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = (ema3 - ema3.shift(1)) / (ema3.shift(1) + 1e-10) * 100
    trix = trix.fillna(0).values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_12h, trix)
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(vol_ma_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        volume_ok = volume > 1.5 * vol_ma_aligned[i]
        
        # Trend filter: price vs 200-day EMA
        uptrend = price > ema200_1d_aligned[i]
        downtrend = price < ema200_1d_aligned[i]
        
        if position == 0:
            # Long conditions: TRIX crosses above zero with volume and uptrend
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and volume_ok and uptrend:
                signals[i] = 0.25
                position = 1
            # Short conditions: TRIX crosses below zero with volume and downtrend
            elif trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and volume_ok and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TRIX returns to zero or below
            if trix_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX returns to zero or above
            if trix_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_TRIX_Signal_Volume_Confirmation_v1"
timeframe = "12h"
leverage = 1.0