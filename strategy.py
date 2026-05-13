#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend
Hypothesis: Donchian channel breakouts with volume confirmation and trend filter
capture strong moves while avoiding false signals. Works in bull markets by 
riding breakouts and in bear markets by catching sharp reversals when price 
breaks below lower band with volume. Designed for low trade frequency (20-40/year)
to minimize fee drag.
"""

name = "4h_Donchian_Breakout_Volume_Trend"
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
    
    # Donchian channel (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Trend filter: 50-period EMA on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG ENTRY: Price breaks above upper Donchian with volume and trend filter
            if close[i] > high_max[i] and volume_confirm[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.30
                position = 1
            # SHORT ENTRY: Price breaks below lower Donchian with volume (no trend filter for shorts in bear markets)
            elif close[i] < low_min[i] and volume_confirm[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to midpoint or breaks below lower band
            midpoint = (high_max[i] + low_min[i]) / 2.0
            if close[i] < midpoint or close[i] < low_min[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price returns to midpoint or breaks above upper band
            midpoint = (high_max[i] + low_min[i]) / 2.0
            if close[i] > midpoint or close[i] > high_max[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals