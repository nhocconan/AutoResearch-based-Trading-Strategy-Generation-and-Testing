#!/usr/bin/env python3
name = "1d_1w_Donchian_Breakout_Momentum"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Donchian channels and momentum
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Donchian channels (20-day) on daily
    high_20 = np.full(len(high_1d), np.nan)
    low_20 = np.full(len(low_1d), np.nan)
    for i in range(20, len(high_1d)):
        high_20[i] = np.max(high_1d[i-20:i])
        low_20[i] = np.min(low_1d[i-20:i])
    
    # Momentum: 10-day ROC on daily close
    roc10 = np.full(len(close_1d), np.nan)
    for i in range(10, len(close_1d)):
        roc10[i] = (close_1d[i] - close_1d[i-10]) / close_1d[i-10]
    
    # Weekly EMA20 for trend filter
    ema20_1w = np.full(len(close_1w), np.nan)
    for i in range(20, len(close_1w)):
        ema20_1w[i] = np.mean(close_1w[i-20:i])
    
    # Align indicators to 1d timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    roc10_aligned = align_htf_to_ltf(prices, df_1d, roc10)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 25  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(roc10_aligned[i]) or np.isnan(ema20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above upper Donchian + positive momentum + weekly uptrend
            if (close[i] > high_20_aligned[i] and 
                roc10_aligned[i] > 0.02 and
                ema20_1w_aligned[i] > ema20_1w_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian + negative momentum + weekly downtrend
            elif (close[i] < low_20_aligned[i] and 
                  roc10_aligned[i] < -0.02 and
                  ema20_1w_aligned[i] < ema20_1w_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Break below lower Donchian or momentum turns negative
            if (close[i] < low_20_aligned[i] or 
                roc10_aligned[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Break above upper Donchian or momentum turns positive
            if (close[i] > high_20_aligned[i] or 
                roc10_aligned[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals