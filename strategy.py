#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_1w_Volume_Filter
Hypothesis: Kaufman's Adaptive Moving Average (KAMA) identifies adaptive trend direction on daily timeframe.
Weekly volume filter (above 20-period average) confirms institutional participation. 
Trades only when KAMA slope aligns with weekly volume strength, reducing whipsaw in low-volume periods.
Works in bull/bear by adapting to volatility and requiring volume confirmation for trend strength.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman's Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0]))[:, None] * np.tril(np.ones((len(close), len(close)))), axis=1)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for volume filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly average volume (20-period)
    volume_1w = df_1w['volume'].values
    vol_ma_1w = np.zeros_like(volume_1w)
    for i in range(len(volume_1w)):
        if i < 19:
            vol_ma_1w[i] = np.nan
        else:
            vol_ma_1w[i] = np.mean(volume_1w[i-19:i+1])
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Daily KAMA for trend
    close_d = prices['close'].values
    kama = calculate_kama(close_d, er_period=10, fast_sc=2, slow_sc=30)
    kama_slope = np.diff(kama, prepend=kama[0])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if np.isnan(vol_ma_1w_aligned[i]) or np.isnan(kama_slope[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly volume filter: current volume > 20-period average
        vol_filter = volume_1w[i // 288] > vol_ma_1w_aligned[i] if i >= 288 else False  # Approximate weekly index
        
        if position == 0:
            # Long: positive KAMA slope + volume confirmation
            if kama_slope[i] > 0 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: negative KAMA slope + volume confirmation
            elif kama_slope[i] < 0 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA slope turns negative
            if kama_slope[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA slope turns positive
            if kama_slope[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_With_1w_Volume_Filter"
timeframe = "1d"
leverage = 1.0