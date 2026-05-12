#!/usr/bin/env python3
"""
1h_Pullback_to_4h_EMA20_with_Volume_Filter
Buys pullbacks to rising 4h EMA20 with volume confirmation and sells on bearish reversal or trend change.
Uses 1d trend filter to avoid counter-trend trades. Designed for low trade frequency.
"""

name = "1h_Pullback_to_4h_EMA20_with_Volume_Filter"
timeframe = "1h"
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
    
    # Volume filter: >1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    # 4h EMA20 for pullback entries
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if (np.isnan(ema_20_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Pullback to rising 4h EMA20 with volume + 1d uptrend
            if (close[i] <= ema_20_4h_aligned[i] * 1.005 and  # within 0.5% of EMA
                ema_20_4h_aligned[i] > ema_20_4h_aligned[i-1] and  # EMA rising
                close[i] > ema_50_1d_aligned[i] and  # above 1d EMA50
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Pullback to falling 4h EMA20 with volume + 1d downtrend
            elif (close[i] >= ema_20_4h_aligned[i] * 0.995 and  # within 0.5% of EMA
                  ema_20_4h_aligned[i] < ema_20_4h_aligned[i-1] and  # EMA falling
                  close[i] < ema_50_1d_aligned[i] and  # below 1d EMA50
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 4h EMA20 OR 1d trend turns down
            if (close[i] < ema_20_4h_aligned[i] * 0.995) or \
               (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above 4h EMA20 OR 1d trend turns up
            if (close[i] > ema_20_4h_aligned[i] * 1.005) or \
               (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals