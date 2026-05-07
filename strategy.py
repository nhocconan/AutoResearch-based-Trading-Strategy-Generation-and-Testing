#!/usr/bin/env python3
"""
12h_KAMA_Direction_With_RSI_And_Volume_Spike
Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) for trend direction, combined with RSI mean reversion and volume spike (>2x average) on 12h timeframe. KAMA adapts to market noise, reducing false signals in choppy markets. RSI identifies overextended conditions for mean-reversion entries. Volume surge confirms momentum behind moves. Designed for low trade frequency (12-37/year) to avoid fee drag, works in both bull and bear markets by adapting trend strength and using mean reversion in extremes.
"""

name = "12h_KAMA_Direction_With_RSI_And_Volume_Spike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    kama_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate ER (Efficiency Ratio) and Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder for correct calc
    
    # Correct volatility calculation: sum of absolute changes over kama_period
    volatility = np.zeros(n)
    for i in range(kama_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-kama_period:i+1])))
    
    # Avoid division by zero
    er = np.zeros(n)
    for i in range(kama_period, n):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    # Smoothing constant
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI calculation
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(1, n):
        if i < rsi_period:
            avg_gain[i] = np.mean(gain[max(0, i-rsi_period+1):i+1]) if i > 0 else 0
            avg_loss[i] = np.mean(loss[max(0, i-rsi_period+1):i+1]) if i > 0 else 0
        else:
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: 20-period average
    vol_ma20 = np.zeros(n)
    for i in range(20, n):
        vol_ma20[i] = np.mean(volume[i-20:i])
    
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, kama_period, rsi_period, 20)  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price below KAMA (dip in uptrend), RSI oversold, volume spike
            if (close[i] < kama[i] and 
                rsi[i] < 30 and 
                vol_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Short: price above KAMA (rally in downtrend), RSI overbought, volume spike
            elif (close[i] > kama[i] and 
                  rsi[i] > 70 and 
                  vol_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses above KAMA or RSI overbought
            if close[i] > kama[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses below KAMA or RSI oversold
            if close[i] < kama[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals