#!/usr/bin/env python3
"""
12h_KAMA_Direction_With_1wTrend_Filter
Hypothesis: KAMA trend direction on 12h with 1-week trend filter and volume confirmation.
Uses KAMA's adaptive smoothing to reduce whipsaw in sideways markets while capturing trends.
Target: 20-30 trades/year to minimize fee drag. Works in bull/bear via weekly trend filter.
"""

name = "12h_KAMA_Direction_With_1wTrend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # For simplicity, we'll compute ER per element in loop later
    # Return arrays for change and volatility for per-bar calculation
    return change, volatility

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    er_length = 10
    fast_sc = 2
    slow_sc = 30
    
    # Pre-calculate change and volatility for ER
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.zeros(n)
    for i in range(1, n):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
        if i >= er_length:
            volatility[i] -= np.abs(close[i-er_length] - close[i-er_length-1])
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    er = np.zeros(n)
    sc = np.zeros(n)
    
    for i in range(1, n):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1-week trend filter: EMA of weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: close above KAMA AND above 1-week EMA with volume confirmation
            if close[i] > kama[i] and close[i] > ema_1w_aligned[i] and volume[i] > vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: close below KAMA AND below 1-week EMA with volume confirmation
            elif close[i] < kama[i] and close[i] < ema_1w_aligned[i] and volume[i] > vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: close crosses below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: close crosses above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals