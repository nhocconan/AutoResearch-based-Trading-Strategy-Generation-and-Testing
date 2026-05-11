#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_with_Volume_and_Chop
Hypothesis: KAMA identifies trend direction, volume confirms momentum, and Choppiness Index filters for trending regimes.
Works in bull/bear by only taking trades in strong trends (Chop < 38.2) and using KAMA crossover for entry.
Designed for low trade frequency (<20/year) on daily timeframe to minimize fee drag.
"""

name = "1d_KAMA_Trend_Filter_with_Volume_and_Chop"
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
    
    # === KAMA (Kaufman Adaptive Moving Average) ===
    # ER (Efficiency Ratio) = abs(close - close[10]) / sum(abs(close - close[-1])) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close(t) - close(t-10)|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # sum of |close(t) - close(t-1)| over 10
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === Choppiness Index (14-period) ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Sum of TR over 14 periods
    atr_sum = np.convolve(tr, np.ones(14), 'full')[:n]  # same as rolling sum
    atr_sum[:13] = np.nan  # first 13 values invalid
    # Highest high and lowest low over 14 periods
    max_high = np.zeros_like(high)
    min_low = np.zeros_like(low)
    for i in range(n):
        if i < 13:
            max_high[i] = np.nan
            min_low[i] = np.nan
        else:
            max_high[i] = np.max(high[i-13:i+1])
            min_low[i] = np.min(low[i-13:i+1])
    # Chop = 100 * log10(sum(tr14) / (max_high - min_low)) / log10(14)
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    chop[:13] = np.nan  # first 13 values invalid
    
    # === Volume Spike (2x 20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 2.0
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers KAMA, Chop, and volume EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(chop[i]) or 
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Only trade in trending markets (Chop < 38.2)
        if chop[i] >= 38.2:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close crosses above KAMA with volume spike
            if close[i] > kama[i] and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Close crosses below KAMA with volume spike
            elif close[i] < kama[i] and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit: Close crosses back through KAMA in opposite direction
            if position == 1:
                if close[i] < kama[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if close[i] > kama[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals