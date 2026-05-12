#!/usr/bin/env python3
name = "1d_KAMA_RSI_Chop"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    er_length = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else np.abs(np.diff(close)).cumsum()
    # Correct volatility calculation: sum of absolute changes over er_length period
    volatility = pd.Series(np.abs(np.diff(close, prepend=close[0]))).rolling(window=er_length, min_periods=1).sum().values
    change_sum = pd.Series(np.abs(np.diff(close, prepend=close[0]))).rolling(window=er_length, min_periods=1).sum().values
    er = np.where(volatility != 0, change_sum / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Chopiness Index (14)
    def true_range(h, l, c_prev):
        return np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    
    tr = true_range(high, low, np.roll(close, 1))
    tr[0] = high[0] - low[0]  # first tr
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.where(atr14 != 0, 100 * np.log10((max_high - min_low) / (atr14 * 14)) / np.log10(14), 50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        if position == 0:
            # Long: KAMA rising, RSI < 50, Chop > 61.8 (range)
            if kama[i] > kama[i-1] and rsi[i] < 50 and chop[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, RSI > 50, Chop > 61.8 (range)
            elif kama[i] < kama[i-1] and rsi[i] > 50 and chop[i] > 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA falling or Chop < 38.2 (trend) or RSI > 70
            if kama[i] < kama[i-1] or chop[i] < 38.2 or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA rising or Chop < 38.2 (trend) or RSI < 30
            if kama[i] > kama[i-1] or chop[i] < 38.2 or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals