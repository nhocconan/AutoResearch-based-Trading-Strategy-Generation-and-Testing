#!/usr/bin/env python3
# 6h_KAMA_Trend_Filter
# Hypothesis: Kaufman Adaptive Moving Average (KAMA) on 1d timeframe provides adaptive trend
# detection that reduces whipsaw in volatile markets. Combined with 6h price position relative
# to KAMA and volume confirmation, this should capture trends while avoiding false signals
# in ranging markets. KAMA adapts to market noise, making it effective in both bull and bear
# regimes. Targets 15-30 trades/year to minimize fee drag.

name = "6h_KAMA_Trend_Filter"
timeframe = "6h"
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
    
    # Get 1d data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Kaufman Adaptive Moving Average (KAMA)
    # Parameters: ER period=10, Fast EMA=2, Slow EMA=30
    er_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0) if len(change) > 0 else np.array([0])
    # Correct volatility calculation: sum of absolute changes over er_period window
    volatility = np.array([np.sum(np.abs(np.diff(close_1d[max(0, i-er_period+1):i+1]))) 
                          for i in range(len(close_1d))])
    er = np.zeros_like(close_1d)
    for i in range(er_period, len(close_1d)):
        if volatility[i] > 0:
            er[i] = np.abs(close_1d[i] - close_1d[i-er_period]) / volatility[i]
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 6h timeframe
    kama_6h = align_htf_to_ltf(prices, df_1d, kama)
    
    # Volume confirmation: volume > 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama_6h[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Volume confirmation
            vol_ok = volume[i] > vol_ma[i]
            
            # Long: price above KAMA + volume confirmation
            if close[i] > kama_6h[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA + volume confirmation
            elif close[i] < kama_6h[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below KAMA
            if close[i] < kama_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above KAMA
            if close[i] > kama_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals