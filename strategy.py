#!/usr/bin/env python3
"""
4h_KAMA_Trend_with_12hEMA_Filter
Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both bull and bear regimes.
12h EMA34 acts as a higher-timeframe trend filter to avoid counter-trend trades.
Enter long when KAMA > 12h EMA34, short when KAMA < 12h EMA34.
Exit when trend reverses. Low-frequency signals to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # KAMA parameters
    er_length = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    # Pad change array to match length
    change = np.concatenate([np.full(er_length, np.nan), change])
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    # Handle NaN in er
    sc = np.where(np.isnan(er), 0, sc)
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[er_length] = close[er_length]  # Seed
    for i in range(er_length + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # 12h EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False).values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(er_length + 1, 34)
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or 
            np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA above 12h EMA34
            if kama[i] > ema_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA below 12h EMA34
            elif kama[i] < ema_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: KAMA crosses below 12h EMA34
            if kama[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: KAMA crosses above 12h EMA34
            if kama[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Trend_with_12hEMA_Filter"
timeframe = "4h"
leverage = 1.0