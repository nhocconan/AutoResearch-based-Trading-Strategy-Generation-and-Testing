#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_Filter
Hypothesis: Use 1-day KAMA to determine trend direction and RSI for entry timing. Long when KAMA turns bullish and RSI crosses above 50, short when KAMA turns bearish and RSI crosses below 50. Exit on opposite RSI cross. Designed to capture medium-term trends with reduced whipsaw in both bull and bear markets.
Target ~10-25 trades/year on 1d by requiring trend confirmation and momentum filter.
"""

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # === KAMA trend indicator (1-day) ===
    close = prices['close'].values
    direction = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * 0.59 + 0.01) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI (14-period) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if indicators not ready
        if np.isnan(kama[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # KAMA trend direction: bullish if close > kama, bearish if close < kama
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        if position == 0:
            # Long: KAMA bullish + RSI crosses above 50
            if kama_bullish and rsi[i] > 50 and rsi[i-1] <= 50:
                signals[i] = 0.25
                position = 1
            # Short: KAMA bearish + RSI crosses below 50
            elif kama_bearish and rsi[i] < 50 and rsi[i-1] >= 50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit on opposite RSI cross
            if position == 1 and rsi[i] < 50 and rsi[i-1] >= 50:
                signals[i] = 0.0
                position = 0
            elif position == -1 and rsi[i] > 50 and rsi[i-1] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_KAMA_Trend_With_RSI_Filter"
timeframe = "1d"
leverage = 1.0