#!/usr/bin/env python3
# 1d_1w_1wKAMA_Trend_Filtered_By_RSI
# Uses weekly KAMA for trend direction and daily RSI for entry timing.
# Long when weekly KAMA rising and daily RSI < 30 (oversold).
# Short when weekly KAMA falling and daily RSI > 70 (overbought).
# Exits when RSI returns to neutral zone (40-60).
# Designed for 1d timeframe to capture medium-term reversals in both bull and bear markets.
# Uses weekly timeframe for trend filter to reduce whipsaw and improve trend accuracy.

name = "1d_1w_1wKAMA_Trend_Filtered_By_RSI"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close)).cumsum()
    volatility = np.diff(volatility, prepend=volatility[0])
    
    er = np.zeros_like(close)
    for i in range(er_period, len(close)):
        if volatility[i-er_period:i].sum() != 0:
            er[i] = change[i-er_period:i].sum() / volatility[i-er_period:i].sum()
        else:
            er[i] = 0
    
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
    
    close = prices['close'].values
    
    # Get weekly data for KAMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly KAMA
    kama_1w = calculate_kama(close_1w, er_period=10, fast_sc=2, slow_sc=30)
    
    # Align KAMA to daily timeframe
    kama_1d = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate daily RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Wilder's smoothing
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    for i in range(15, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN
        if np.isnan(kama_1d[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: weekly KAMA rising and daily RSI oversold (< 30)
            if kama_1d[i] > kama_1d[i-1] and rsi[i] < 30:
                signals[i] = 0.25
                position = 1
            # Short: weekly KAMA falling and daily RSI overbought (> 70)
            elif kama_1d[i] < kama_1d[i-1] and rsi[i] > 70:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when RSI returns to neutral (>= 40)
            if rsi[i] >= 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when RSI returns to neutral (<= 60)
            if rsi[i] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals