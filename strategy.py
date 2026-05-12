#!/usr/bin/env python3
"""
12h_1d_Donchian20_Breakout_Volume_Trend
Hypothesis: Daily Donchian channel (20-period) breakouts on 12h timeframe capture significant
trend moves with high probability. Volume confirmation filters false breakouts. Trend filter
using 12h EMA(50) ensures trades align with intermediate-term direction. Works in bull/bear
markets by following momentum. Target: 50-150 trades over 4 years.
"""
name = "12h_1d_Donchian20_Breakout_Volume_Trend"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DONCHIAN CHANNEL (20-period) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period Donchian channels on daily data
    high_roll = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    upper_donchian = high_roll
    lower_donchian = low_roll
    
    # Align daily Donchian levels to 12h timeframe
    upper_donchian_aligned = align_htf_to_ltf(prices, df_1d, upper_donchian)
    lower_donchian_aligned = align_htf_to_ltf(prices, df_1d, lower_donchian)
    
    # === VOLUME CONFIRMATION (20-period on 12h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)  # Volume 1.5x average
    
    # === TREND FILTER: 12h EMA(50) ===
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # For EMA(50) and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_donchian_aligned[i]) or np.isnan(lower_donchian_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Donchian with volume and above EMA50
            if (high[i] > upper_donchian_aligned[i] and 
                close[i] > upper_donchian_aligned[i] and
                volume_spike[i] and
                close[i] > ema_50[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian with volume and below EMA50
            elif (low[i] < lower_donchian_aligned[i] and 
                  close[i] < lower_donchian_aligned[i] and
                  volume_spike[i] and
                  close[i] < ema_50[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below lower Donchian or reverses with volume
            if low[i] < lower_donchian_aligned[i] and volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Donchian or reverses with volume
            if high[i] > upper_donchian_aligned[i] and volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals