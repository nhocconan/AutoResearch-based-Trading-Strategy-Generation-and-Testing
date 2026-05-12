#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_Filter
Hypothesis: Donchian channel breakouts on 4h capture medium-term trends, while daily EMA200 trend filter ensures alignment with higher timeframe momentum. Volume confirmation (1.5x average) filters false breakouts. This combination should work in both bull and bear markets by capturing strong moves while avoiding whipsaws. Target: 20-40 trades/year per symbol.
"""

name = "4h_Donchian20_Breakout_1dTrend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Daily data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Donchian channel on 4h (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if np.isnan(ema_200_1d_aligned[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high + above daily EMA200 + volume spike
            if (close[i] > high_max[i] and 
                close[i] > ema_200_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + below daily EMA200 + volume spike
            elif (close[i] < low_min[i] and 
                  close[i] < ema_200_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low OR closes below daily EMA200
            if (close[i] < low_min[i]) or \
               (close[i] < ema_200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high OR closes above daily EMA200
            if (close[i] > high_max[i]) or \
               (close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals