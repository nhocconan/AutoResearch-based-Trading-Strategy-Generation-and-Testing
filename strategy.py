#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_Volume_Confirmation
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing a robust trend signal.
Combined with volume confirmation and position sizing of 0.25, this strategy aims to capture trends
while minimizing false signals in choppy markets. Target: 20-30 trades/year to reduce fee drag.
"""

name = "4h_KAMA_Trend_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    def kama(close, period=10, fast=2, slow=30):
        change = np.abs(np.subtract(close[period:], close[:-period]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
        sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
        kama = np.zeros_like(close)
        kama[period] = close[period]
        for i in range(period+1, len(close)):
            kama[i] = kama[i-1] + sc[i-period] * (close[i-1] - kama[i-1])
        return kama
    
    # Calculate KAMA with period=10
    kama_values = kama(close, period=10, fast=2, slow=30)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        if position == 0:
            # LONG: Price above KAMA with volume confirmation
            if close[i] > kama_values[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA with volume confirmation
            elif close[i] < kama_values[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA
            if close[i] < kama_values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA
            if close[i] > kama_values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals