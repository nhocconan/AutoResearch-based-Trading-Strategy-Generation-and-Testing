#!/usr/bin/env python3
"""
6h_21_EMA_Crossover_With_Volume_Filter
Simple 21-period EMA crossover on 6h timeframe with volume confirmation.
Long when price crosses above EMA21 with volume > 1.5x average, short when crosses below.
Uses volume filter to avoid whipsaws in sideways markets.
Designed for low trade frequency (~50-150 total trades over 4 years) to minimize fee drag.
Works in both bull and bear markets by following trend with volume confirmation.
"""

name = "6h_21_EMA_Crossover_With_Volume_Filter"
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
    
    # Volume spike: >1.5x 20-period average (on 6h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # 6h EMA21 for trend
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(21, n):
        if np.isnan(ema_21[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price crosses above EMA21 with volume spike
            if (close[i] > ema_21[i] and 
                close[i-1] <= ema_21[i-1] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below EMA21 with volume spike
            elif (close[i] < ema_21[i] and 
                  close[i-1] >= ema_21[i-1] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below EMA21
            if close[i] < ema_21[i] and close[i-1] >= ema_21[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above EMA21
            if close[i] > ema_21[i] and close[i-1] <= ema_21[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals