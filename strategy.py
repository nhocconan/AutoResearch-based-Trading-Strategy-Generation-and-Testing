#!/usr/bin/env python3
"""
6h_1d_200ma_volume_momentum
Uses 200-period moving average on 1d for trend direction and volume momentum on 6s for entry timing.
Long when price > 200ma and volume > 1.5x 20-period average, short when price < 200ma and volume > 1.5x average.
Exit when price crosses back below/above 200ma.
Designed for low trade frequency (target: 10-30 trades/year) to minimize fee drift.
Works in both trending and ranging markets by combining long-term trend with short-term momentum.
"""

name = "6h_1d_200ma_volume_momentum"
timeframe = "6h"
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
    
    # Get 1d data for 200-period MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 200-period MA on 1d
    ma_200 = pd.Series(close_1d).rolling(window=200, min_periods=200).mean().values
    ma_200_aligned = align_htf_to_ltf(prices, df_1d, ma_200)
    
    # Volume momentum: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_momentum = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if np.isnan(ma_200_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Long entry: price > 200ma and volume momentum
        if close[i] > ma_200_aligned[i] and vol_momentum[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short entry: price < 200ma and volume momentum
        elif close[i] < ma_200_aligned[i] and vol_momentum[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price crosses back through 200ma
        elif position == 1 and close[i] <= ma_200_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= ma_200_aligned[i]:
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