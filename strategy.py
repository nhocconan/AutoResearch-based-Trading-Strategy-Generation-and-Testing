#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyTrend_Filter
Hypothesis: Donchian(20) breakouts filtered by weekly trend (close > weekly SMA50 = uptrend, close < weekly SMA50 = downtrend) capture strong directional moves while avoiding counter-trend whipsaws. Weekly trend filter ensures alignment with higher timeframe momentum, reducing false breakouts in sideways markets. Volume confirmation (1.5x average) adds conviction. Designed for 6-12 trades per year per symbol with clear trend-following edge in both bull and bear markets.
"""

name = "6h_Donchian20_WeeklyTrend_Filter"
timeframe = "6h"
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
    
    # Volume spike: >1.5x 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly SMA50 for trend filter
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    
    # Donchian(20) channels on 6h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align weekly trend to 6h timeframe
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(sma_50_1w_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian(20) high + weekly uptrend + volume spike
            if (close[i] > highest_high[i] and 
                close[i] > sma_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian(20) low + weekly downtrend + volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < sma_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian(20) low OR closes below weekly SMA50
            if (close[i] < lowest_low[i]) or \
               (close[i] < sma_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian(20) high OR closes above weekly SMA50
            if (close[i] > highest_high[i]) or \
               (close[i] > sma_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals