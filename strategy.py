#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_Pullback
Hypothesis: 1-day KAMA trend with RSI pullback entries.
KAMA adapts to market noise, reducing whipsaws in sideways markets.
RSI pullback (30-40 for long, 60-70 for short) enters during retracements in the trend.
Designed for low trade frequency (target: 10-25 trades/year) to minimize fee drag.
Works in bull markets via trend following and in bear markets via short entries on pullbacks.
"""

name = "1d_KAMA_Trend_RSI_Pullback"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER (Efficiency Ratio) = |change| / sum(|changes|)
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close))
    er = np.zeros_like(change)
    for i in range(2, len(change)):
        if np.sum(abs_change[i-9:i+1]) > 0:
            er[i] = np.abs(change[i]) / np.sum(abs_change[i-9:i+1])
        else:
            er[i] = 0
    # Smooth ER with constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for KAMA and RSI
    start_idx = 30
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA and RSI pulling back from oversold (30-40)
            if close[i] > kama[i] and 30 <= rsi[i] <= 40:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA and RSI pulling back from overbought (60-70)
            elif close[i] < kama[i] and 60 <= rsi[i] <= 70:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA or RSI overbought (>70)
            if close[i] < kama[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA or RSI oversold (<30)
            if close[i] > kama[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals