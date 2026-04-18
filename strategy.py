#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_RSI_Filter
Hypothesis: KAMA adapts to market efficiency, reducing whipsaw in choppy markets. Combined with RSI > 50 for long and RSI < 50 for short, it captures directional moves while avoiding false signals. Works in both bull and bear markets by following the trend with adaptive smoothing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # KAMA calculation
    def calculate_kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.zeros_like(close)
        for i in range(length, len(close)):
            if volatility[i] != 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        # Smoothing Constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, length=10, fast=2, slow=30)
    
    # RSI calculation
    def calculate_rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=length, min_periods=length).mean().values
        avg_loss = pd.Series(loss).rolling(window=length, min_periods=length).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, length=14)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30  # Need KAMA and RSI warmup
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Long: price > KAMA and RSI > 50
            if price > kama_val and rsi_val > 50:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA and RSI < 50
            elif price < kama_val and rsi_val < 50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price < KAMA or RSI < 50
            if price < kama_val or rsi_val < 50:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > KAMA or RSI > 50
            if price > kama_val or rsi_val > 50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Trend_With_RSI_Filter"
timeframe = "4h"
leverage = 1.0