#!/usr/bin/env python3
# 4h_KAMA_Trend_With_Volume_Confirmation
# Hypothesis: KAMA adapts to market efficiency, reducing whipsaw in choppy markets while capturing trends.
# Combined with volume confirmation (>1.5x 20-period average) to filter low-conviction moves.
# Designed for 4h timeframe targeting 20-50 trades/year (80-200 total over 4 years).
# Works in bull/bear via adaptive trend following and volume filter.

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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAUFMAN'S ADAPTIVE MOVING AVERAGE (KAMA)
    def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
        # Efficiency Ratio
        change = np.abs(close - np.roll(close, er_period))
        volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
        # Handle first er_period values
        for i in range(er_period):
            volatility[i] = np.sum(np.abs(np.diff(close[:i+1], prepend=close[0])))
        er = np.where(volatility != 0, change / volatility, 0)
        
        # Smoothing Constants
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        
        # KAMA calculation
        kama = np.full_like(close, np.nan)
        kama[0] = close[0]
        for i in range(1, len(close)):
            if not np.isnan(kama[i-1]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
            else:
                kama[i] = close[i]
        return kama
    
    # Calculate KAMA on close prices
    kama = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(kama[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above KAMA with volume confirmation
            if close[i] > kama[i] and close[i-1] <= kama[i-1] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA with volume confirmation
            elif close[i] < kama[i] and close[i-1] >= kama[i-1] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below KAMA
            if close[i] < kama[i] and close[i-1] >= kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above KAMA
            if close[i] > kama[i] and close[i-1] <= kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals