#!/usr/bin/env python3
# 4h_KAMA_Trend_With_Volume_Filter_v1
# Hypothesis: KAMA identifies adaptive trend direction; long when price > KAMA, short when price < KAMA.
# Volume filter (volume > 1.5x 20-period average) ensures momentum confirmation.
# Uses 4h timeframe to balance trade frequency and capture sustained moves.
# Designed for both bull and bear markets by following adaptive trend with volume confirmation.

name = "4h_KAMA_Trend_With_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd

def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman's Adaptive Moving Average."""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    
    # Calculate efficiency ratio
    er = np.zeros_like(close)
    for i in range(len(close)):
        if i >= er_length:
            price_change = np.abs(close[i] - close[i-er_length])
            sum_volatility = np.sum(volatility[i-er_length+1:i+1])
            if sum_volatility > 0:
                er[i] = price_change / sum_volatility
            else:
                er[i] = 0
    
    # Calculate smoothing constant
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1))**2
    
    # Calculate KAMA
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
    
    # Calculate KAMA for trend identification
    kama = calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        if position == 0:
            # LONG: Price above KAMA with volume confirmation
            if close[i] > kama[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA with volume confirmation
            elif close[i] < kama[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals