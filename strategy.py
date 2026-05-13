#!/usr/bin/env python3
"""
4h_SuperTrend_Retest_Entry
Hypothesis: SuperTrend (ATR=10, mult=3) defines the trend, and retests of the SuperTrend line during pullbacks provide high-probability entries in the direction of the trend. Volume confirmation filters out weak moves. Works in both bull and bear markets by following the trend direction. Target: 20-40 trades/year to minimize fee drag.
"""

name = "4h_SuperTrend_Retest_Entry"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(10)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # SuperTrend calculation
    upper = (high + low) / 2 + 3 * atr
    lower = (high + low) / 2 - 3 * atr
    
    supertrend = np.zeros_like(close)
    direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper[0]
    direction[0] = 1
    
    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            direction[i] = 1
        elif close[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            supertrend[i] = max(lower[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper[i], supertrend[i-1])
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: retest of SuperTrend support in uptrend with volume confirmation
            if direction[i] == 1 and close[i] > supertrend[i] and close[i] <= supertrend[i] + 0.5 * atr[i] and volume_confirmed[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: retest of SuperTrend resistance in downtrend with volume confirmation
            elif direction[i] == -1 and close[i] < supertrend[i] and close[i] >= supertrend[i] - 0.5 * atr[i] and volume_confirmed[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: close below SuperTrend
            if close[i] < supertrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: close above SuperTrend
            if close[i] > supertrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals