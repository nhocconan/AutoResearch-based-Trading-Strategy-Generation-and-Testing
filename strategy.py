#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_1d_RSI_Filter_v1
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) on 4h for trend direction, filtered by 1d RSI to avoid extremes. Enter long when KAMA turns up and RSI < 70, short when KAMA turns down and RSI > 30. Designed for 20-40 trades/year to minimize fee drift. Works in bull via trend following and bear via mean-reversion filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_len=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
    # Avoid division by zero
    volatility = np.where(volatility == 0, 1, volatility)
    er = change / volatility
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
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
    
    # 1d RSI for filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    rsi_period = 14
    delta = np.diff(df_1d['close'].values, prepend=df_1d['close'].values[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # 4h KAMA for trend
    kama = calculate_kama(close, er_len=10, fast=2, slow=30)
    kama_diff = np.diff(kama, prepend=kama[0])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    for i in range(1, n):
        if np.isnan(rsi_aligned[i]) or np.isnan(kama[i]):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi_aligned[i]
        kama_val = kama[i]
        kama_prev = kama[i-1]
        
        if position == 0:
            # Long: KAMA turning up AND RSI not overbought
            if kama_val > kama_prev and rsi_val < 70:
                signals[i] = size
                position = 1
            # Short: KAMA turning down AND RSI not oversold
            elif kama_val < kama_prev and rsi_val > 30:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: KAMA turning down OR RSI overbought
            if kama_val < kama_prev or rsi_val >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: KAMA turning up OR RSI oversold
            if kama_val > kama_prev or rsi_val <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_KAMA_Trend_With_1d_RSI_Filter_v1"
timeframe = "4h"
leverage = 1.0