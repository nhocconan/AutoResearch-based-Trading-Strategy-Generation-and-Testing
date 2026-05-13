#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_Filter
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing a smooth trend line that reduces whipsaws. 
Combined with RSI(14) for momentum confirmation and a volatility filter (ATR-based), this strategy aims to capture strong trends 
while avoiding choppy markets. Designed for low trade frequency (~10-25/year) on daily timeframe to minimize fee drag.
"""

name = "1d_KAMA_Trend_With_RSI_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    kama_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.zeros(n)
    for i in range(kama_period, n):
        er[i] = np.sum(change[i-kama_period+1:i+1]) / np.sum(volatility[i-kama_period+1:i+1]) if np.sum(volatility[i-kama_period+1:i+1]) > 0 else 0
    
    # Smoothing constants
    sc = np.zeros(n)
    fast_sc = 2 / (fast_ema + 1)
    slow_sc = 2 / (slow_ema + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # ATR(14) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volatility filter: ATR ratio (current ATR / 50-period average ATR)
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr > (0.5 * atr_ma)  # Avoid extremely low volatility
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Price above KAMA, RSI > 50 (bullish momentum), volatility filter passed
            if close[i] > kama[i] and rsi[i] > 50 and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA, RSI < 50 (bearish momentum), volatility filter passed
            elif close[i] < kama[i] and rsi[i] < 50 and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA OR RSI < 40 (loss of momentum)
            if close[i] < kama[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA OR RSI > 60 (loss of momentum)
            if close[i] > kama[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals