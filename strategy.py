#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_TrendFilter_Volume
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) direction as primary trend filter, combined with RSI momentum and volume confirmation. KAMA adapts to market noise, reducing whipsaws in ranging markets while capturing trends. RSI (14) avoids overbought/oversold extremes, and volume surge confirms institutional participation. Designed for 4h timeframe to target 20-50 trades/year, balancing signal quality with low frequency to overcome fee drag in both bull and bear markets.
"""

name = "4h_KAMA_Direction_RSI_TrendFilter_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (2-period ER, 30-period smoothing)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume moving average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup for KAMA/RSI/volume
        # Skip if any data is NaN
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: KAMA upward (close > KAMA), RSI not overbought (<60), volume above average
            if close[i] > kama[i] and rsi[i] < 60 and volume[i] > vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA downward (close < KAMA), RSI not oversold (>40), volume above average
            elif close[i] < kama[i] and rsi[i] > 40 and volume[i] > vol_ma[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns downward or RSI overbought
            if close[i] < kama[i] or rsi[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns upward or RSI oversold
            if close[i] > kama[i] or rsi[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals