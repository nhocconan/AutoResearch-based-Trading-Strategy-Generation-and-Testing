#!/usr/bin/env python3
"""
12h_KAMA_Trend_Filter_Volume
Hypothesis: Combines Kaufman Adaptive Moving Average (KAMA) for trend direction with volume confirmation on 12h timeframe. 
KAMA adapts to market noise, reducing whipsaws in sideways markets while capturing trends. Volume filter ensures 
trades occur with institutional interest. Targets 12-37 trades/year with low fee drift. Works in bull/bear via 
adaptive trend strength.
"""

name = "12h_KAMA_Trend_Filter_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for KAMA calculation (using same timeframe as strategy)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Efficiency Ratio and Smoothing Constants for KAMA
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None
    
    # Proper ER calculation over 10 periods
    er = np.zeros(n)
    for i in range(10, n):
        price_change = np.abs(close[i] - close[i-10])
        price_volatility = np.sum(np.abs(np.diff(close[i-10:i+1])))
        if price_volatility > 0:
            er[i] = price_change / price_volatility
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Volume filter: current volume > 1.3x 20-period EMA of volume
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after KAMA warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if KAMA or volume data invalid
        if np.isnan(kama[i]) or np.isnan(vol_ema20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction: price vs KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        if position == 0:
            # Enter long: price above KAMA with volume confirmation
            if price_above_kama and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA with volume confirmation
            elif price_below_kama and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA
            if not price_above_kama:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above KAMA
            if not price_below_kama:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals