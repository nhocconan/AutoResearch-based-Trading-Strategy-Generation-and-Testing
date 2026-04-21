# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_1w_Volume_Filter
Hypothesis: 1-day KAMA trend filter combined with 1-week volume confirmation for position sizing.
KAMA adapts to market noise, reducing false signals in choppy conditions. Weekly volume filter ensures
trades occur during institutional participation periods. Designed for 1d timeframe to target 10-30 trades/year.
Works in bull markets by following adaptive trend and in bear markets by avoiding false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    vol = np.abs(np.diff(close, prepend=close[0])).sum()  # placeholder, will be calculated properly
    
    # Proper efficiency ratio calculation
    er = np.zeros_like(close)
    for i in range(er_length, len(close)):
        if i >= er_length:
            direction = np.abs(close[i] - close[i-er_length])
            volatility = np.sum(np.abs(np.diff(close[i-er_length:i+1])))
            if volatility != 0:
                er[i] = direction / volatility
            else:
                er[i] = 0
    
    # Smoothing constants
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1))**2
    
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for KAMA and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Load 1w data for volume filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1-day KAMA
    close_1d = df_1d['close'].values
    kama_1d = calculate_kama(close_1d, er_length=10, fast_sc=2, slow_sc=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 1-week average volume
    volume_1w = df_1w['volume'].values
    vol_ma_1w = np.zeros_like(volume_1w)
    for i in range(len(volume_1w)):
        if i >= 10:
            vol_ma_1w[i] = np.mean(volume_1w[max(0, i-9):i+1])
        else:
            vol_ma_1w[i] = np.mean(volume_1w[:i+1]) if i > 0 else volume_1w[i]
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    
    for i in range(30, n):
        # Skip if indicators not ready
        if np.isnan(kama_1d_aligned[i]) or np.isnan(vol_ma_1w_aligned[i]):
            continue
            
        price = prices['close'].iloc[i]
        volume = df_1w['volume'].iloc[min(i // (24*7), len(df_1w)-1)] if i >= 30 else volume_1w[0]
        vol_ma = vol_ma_1w_aligned[i]
        
        # Volume filter: current weekly volume > 1.5 * 10-period average
        volume_ok = volume > 1.5 * vol_ma if vol_ma > 0 else False
        
        if volume_ok:
            # Long when price above KAMA
            if price > kama_1d_aligned[i]:
                signals[i] = 0.25
            # Short when price below KAMA
            elif price < kama_1d_aligned[i]:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_KAMA_Trend_With_1w_Volume_Filter"
timeframe = "1d"
leverage = 1.0