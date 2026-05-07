# -*- coding: utf-8 -*-
#!/usr/bin/env python3
name = "6h_Donchian20_WeeklyPivot_Dir_Volume"
timeframe = "6h"
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
    
    # Weekly pivot direction (from weekly OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Bullish bias: weekly close > weekly open
    weekly_bullish = weekly_close > weekly_open
    pivot_bullish = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    pivot_bearish = ~pivot_bullish  # bearish bias
    
    # Daily Donchian(20) breakout levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    donch_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Volume spike: current volume > 2.0 x 24-period average (6h * 24 = 6 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Ensure volume MA data
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(pivot_bullish[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Break above daily Donchian high in weekly bullish bias with volume spike
            if close[i] > donch_high_aligned[i] and pivot_bullish[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below daily Donchian low in weekly bearish bias with volume spike
            elif close[i] < donch_low_aligned[i] and pivot_bearish[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Price returns to opposite Donchian level
            if position == 1 and close[i] < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals