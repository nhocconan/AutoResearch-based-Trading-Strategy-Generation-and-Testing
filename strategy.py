#!/usr/bin/env python3
"""
4H_KAMA_Trend_Filter_Volume_Signal
Hypothesis: Uses KAMA (Kaufman Adaptive Moving Average) on 4h to detect trend direction, 
confirmed by RSI momentum and volume spike. Designed for 4h timeframe to capture 
trend continuation with low trade frequency (target: 20-40 trades/year). 
Works in both bull and bear markets by following KAMA trend direction, avoiding 
counter-trend trades. Uses discrete position sizing (0.25) to minimize fee churn.
"""

name = "4H_KAMA_Trend_Filter_Volume_Signal"
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
    
    # Calculate KAMA on 4h close
    # Efficiency Ratio (ER) = |close - close[10]| / sum(|close - close[-1]|) over 10 periods
    change = np.abs(close - np.roll(close, 10))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)  # placeholder, will compute properly below
    
    # Proper ER calculation
    change_10 = np.abs(np.subtract(close, np.roll(close, 10)))
    volatility_10 = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility_10[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
    
    er = np.divide(change_10, volatility_10, out=np.full_like(change_10, 0.0), where=volatility_10!=0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # start after 10 periods
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) for momentum confirmation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, 0.0), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Warmup for KAMA and RSI
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        if position == 0:
            # Long entry: price above KAMA + RSI > 50 + volume spike
            if (price_above_kama and 
                rsi[i] > 50 and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price below KAMA + RSI < 50 + volume spike
            elif (price_below_kama and 
                  rsi[i] < 50 and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below KAMA or RSI < 40
            if (close[i] < kama[i] or rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above KAMA or RSI > 60
            if (close[i] > kama[i] or rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals