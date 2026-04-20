#!/usr/bin/env python3
"""
6h_RSI_Stochastic_Combo_With_Volume_Filter
Hypothesis: Combine RSI (oversold/overbought) with Stochastic (momentum) and volume confirmation on 6h timeframe. 
Long when RSI<30, Stochastic K<20, and volume>1.5x average; short when RSI>70, Stochastic K>80, and volume>1.5x average.
Exit when RSI crosses 50 or volume drops below average. Uses volume filter to avoid false signals in low-volume periods.
Works in bull/bear: momentum extremes with volume confirmation capture reversals; volume filter reduces whipsaw.
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
"""

name = "6h_RSI_Stochastic_Combo_With_Volume_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    rsi = np.zeros(n)
    
    # Wilder's smoothing
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100
    
    # Calculate Stochastic Oscillator (14,3,3)
    lowest_low = np.zeros(n)
    highest_high = np.zeros(n)
    
    for i in range(n):
        if i < 13:
            lowest_low[i] = np.nan
            highest_high[i] = np.nan
        else:
            lowest_low[i] = np.min(low[i-13:i+1])
            highest_high[i] = np.max(high[i-13:i+1])
    
    stoch_k = np.zeros(n)
    for i in range(n):
        if highest_high[i] == lowest_low[i] or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            stoch_k[i] = 50
        else:
            stoch_k[i] = 100 * (close[i] - lowest_low[i]) / (highest_high[i] - lowest_low[i])
    
    # Smooth %K to get %D (3-period SMA of %K)
    stoch_d = np.zeros(n)
    for i in range(n):
        if i < 2:
            stoch_d[i] = np.nan
        else:
            stoch_d[i] = np.mean(stoch_k[i-2:i+1])
    
    # Volume average (20-period)
    vol_ma = np.zeros(n)
    for i in range(n):
        if i < 19:
            vol_ma[i] = np.nan
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(stoch_k[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI<30, Stochastic K<20, volume>1.5x average
            if rsi[i] < 30 and stoch_k[i] < 20 and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI>70, Stochastic K>80, volume>1.5x average
            elif rsi[i] > 70 and stoch_k[i] > 80 and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses above 50 OR volume drops below average
            if rsi[i] > 50 or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI crosses below 50 OR volume drops below average
            if rsi[i] < 50 or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals