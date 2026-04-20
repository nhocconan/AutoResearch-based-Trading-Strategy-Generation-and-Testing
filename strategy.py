#!/usr/bin/env python3
# 1d_1w_KAMA_Trend_RSI_Momentum_v2
# Hypothesis: Use weekly KAMA trend direction with daily RSI momentum on pullbacks.
# Only trade in direction of weekly trend during pullbacks (RSI < 40 for longs, > 60 for shorts).
# Weekly trend filter reduces whipsaw in sideways markets, RSI captures momentum.
# Designed for low trade frequency (10-25/year) to minimize fee drag.

name = "1d_1w_KAMA_Trend_RSI_Momentum_v2"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
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
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly KAMA for trend
    close_1w = df_1w['close'].values
    kama_1w = calculate_kama(close_1w, er_length=10, fast_sc=2, slow_sc=30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Daily RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: weekly uptrend + RSI oversold bounce
            if close[i] > kama_1w_aligned[i] and rsi[i] < 40:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + RSI overbought bounce
            elif close[i] < kama_1w_aligned[i] and rsi[i] > 60:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: weekly trend turns down or RSI overbought
            if close[i] < kama_1w_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: weekly trend turns up or RSI oversold
            if close[i] > kama_1w_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals