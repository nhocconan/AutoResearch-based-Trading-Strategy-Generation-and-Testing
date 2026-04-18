#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter
Hypothesis: Uses KAMA(10) to capture trend direction on daily timeframe. 
Enters long when price > KAMA and volume > 1.5x 20-day average, short when price < KAMA with volume confirmation.
Designed for low trade frequency (~10-20/year) with trend-following capability in both bull and bear markets.
KAMA adapts to market noise, reducing whipsaws in choppy conditions while capturing sustained trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA(10) calculation
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, 10))  # |close[i] - close[i-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |diff| over 10 periods
    
    # Handle array dimensions
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    
    # ER = change / volatility, handle division by zero
    er = np.divide(change, volatility, out=np.full_like(change, np.nan), where=volatility!=0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # for 2-period EMA
    slow_sc = 2 / (30 + 1)  # for 30-period EMA
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > KAMA with volume spike
            if close[i] > kama[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA with volume spike
            elif close[i] < kama[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price < KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price > KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_Filter"
timeframe = "1d"
leverage = 1.0