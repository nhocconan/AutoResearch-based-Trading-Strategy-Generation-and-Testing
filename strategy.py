#!/usr/bin/env python3
"""
EXPERIMENT #002 - HMA Crossover(4h) Trend Following Strategy
=============================================================
Hypothesis: Hull Moving Average (HMA) reduces lag compared to EMA while maintaining
smoothness. HMA(16)/HMA(48) crossover on 4h should capture trends faster than EMA(21/55)
baseline, potentially improving Sharpe ratio. HMA's weighted calculation responds quicker
to price changes while filtering noise better than simple MA crossovers.

Key differences from baseline:
- HMA reduces lag vs EMA (faster trend detection)
- Same 4h timeframe as Supertrend #001 (cleaner trends)
- Discrete signal levels (0.0, ±0.35) to minimize churning costs
- Conservative 35% position sizing for drawdown control
"""

import numpy as np
import pandas as pd

name = "hma_crossover_4h_v1"
timeframe = "4h"
leverage = 1.0


def wma(x, n):
    """Calculate Weighted Moving Average"""
    weights = np.arange(1, n + 1)
    wma_val = np.convolve(x, weights / weights.sum(), mode='valid')
    # Pad beginning with NaN to match input length
    padding = np.full(n - 1, np.nan)
    return np.concatenate([padding, wma_val])


def hma(x, n):
    """Calculate Hull Moving Average"""
    if len(x) < n:
        return np.full(len(x), np.nan)
    
    # WMA(n/2)
    wma_half = wma(x, n // 2)
    # WMA(n)
    wma_full = wma(x, n)
    
    # 2*WMA(n/2) - WMA(n)
    diff = 2 * wma_half - wma_full
    
    # WMA of diff with sqrt(n)
    sqrt_n = int(np.sqrt(n))
    if sqrt_n < 1:
        sqrt_n = 1
    
    hma_val = wma(diff, sqrt_n)
    return hma_val


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    n = len(close)
    
    if n < 100:
        return np.zeros(n)
    
    # Calculate HMA(16) and HMA(48)
    hma_fast = hma(close, 16)
    hma_slow = hma(close, 48)
    
    # Determine crossover signals
    signals = np.zeros(n)
    SIZE = 0.35  # 35% position size - critical for drawdown control
    
    # Track previous signal to avoid excessive churning
    prev_signal = 0.0
    
    for i in range(n):
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            signals[i] = 0.0
            continue
        
        # Generate raw signal based on crossover
        if hma_fast[i] > hma_slow[i]:
            raw_signal = SIZE
        elif hma_fast[i] < hma_slow[i]:
            raw_signal = -SIZE
        else:
            raw_signal = 0.0
        
        # Only change signal if there's a meaningful crossover
        # This reduces churning and fees
        if raw_signal != prev_signal:
            # Confirm signal persists for at least 1 bar (already satisfied by using current bar)
            signals[i] = raw_signal
            prev_signal = raw_signal
        else:
            signals[i] = prev_signal
    
    return signals