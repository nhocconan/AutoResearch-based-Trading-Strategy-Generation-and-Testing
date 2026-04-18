#!/usr/bin/env python3
"""
12h_KAMA_Trend_Filter_V1
12h strategy using Kaufman's Adaptive Moving Average (KAMA) for trend filtering.
- Long: Price above KAMA(10,2,30) + rising KAMA slope + volume > 1.5x average
- Short: Price below KAMA(10,2,30) + falling KAMA slope + volume > 1.5x average
- Exit: Opposite signal
Designed for ~12-25 trades/year per symbol (48-100 total over 4 years)
Adapts to market conditions: fast in trends, slow in ranging markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA(10,2,30) on close
    # ER (Efficiency Ratio) = |change over 10 periods| / sum of absolute changes over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    # For t < 10, we need to handle boundary
    change_full = np.full(n, np.nan)
    change_full[10:] = change
    
    # Sum of absolute changes over 10 periods
    abs_changes = np.abs(np.diff(close, n=1))
    abs_sum = np.full(n, np.nan)
    for i in range(10, n):
        abs_sum[i] = np.sum(abs_changes[i-9:i+1])  # sum of last 10 absolute changes
    
    # ER = change / abs_sum, handle division by zero
    er = np.full(n, np.nan)
    mask = (abs_sum > 0) & (~np.isnan(abs_sum))
    er[mask] = change_full[mask] / abs_sum[mask]
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # for EMA(2)
    slow_sc = 2 / (30 + 1)  # for EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[0] = close[0]  # start with first price
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA slope (direction) - 3-period slope
    kama_slope = np.full(n, np.nan)
    for i in range(3, n):
        if not np.isnan(kama[i]) and not np.isnan(kama[i-3]):
            kama_slope[i] = (kama[i] - kama[i-3]) / 3
    
    # Volume filter: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need sufficient lookback for KAMA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(kama_slope[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        kama_rising = kama_slope[i] > 0
        kama_falling = kama_slope[i] < 0
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price above KAMA + rising KAMA + volume filter
            if price_above_kama and kama_rising and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA + falling KAMA + volume filter
            elif price_below_kama and kama_falling and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below KAMA
            if price_below_kama:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above KAMA
            if price_above_kama:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Trend_Filter_V1"
timeframe = "12h"
leverage = 1.0