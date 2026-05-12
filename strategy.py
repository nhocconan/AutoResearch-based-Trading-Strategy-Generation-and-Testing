#!/usr/bin/env python3
"""
1D_KAMA_TREND_RSI_WITH_VOLUME_CONFIRMATION
Hypothesis: Use KAMA for trend direction, RSI for momentum, and volume spike for confirmation on daily timeframe.
Designed to work in both bull and bear markets by filtering trades with volume confirmation and trend alignment.
Target: 15-25 trades/year to stay under 150 total trades limit for 1d timeframe.
KAMA adapts to market noise, reducing false signals in choppy conditions.
"""
name = "1D_KAMA_TREND_RSI_WITH_VOLUME_CONFIRMATION"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA trend (ER=10, FAST=2, SLOW=30)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # will fix below
    # Recompute volatility properly: sum of absolute changes over ER period
    er = np.zeros(n)
    for i in range(n):
        if i < 10:
            er[i] = 0
        else:
            change_sum = np.sum(np.abs(np.diff(close[i-9:i+1])))
            volatility_sum = np.sum(np.abs(np.diff(close[i-9:i+1])))
            er[i] = change_sum / (volatility_sum + 1e-10)
    ss = np.power(er * (2/2 - 2/30) + 2/30, 2)  # smoothing constant
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + ss[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after RSI warmup
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above KAMA (uptrend), RSI > 50 (bullish momentum), volume spike
            if (close[i] > kama[i] and 
                rsi[i] > 50 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA (downtrend), RSI < 50 (bearish momentum), volume spike
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA OR RSI < 40 (loss of momentum)
            if close[i] < kama[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA OR RSI > 60 (loss of momentum)
            if close[i] > kama[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals