#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_200MA_Filter
Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both trending and ranging markets. Combined with 200-period MA filter to avoid false signals during strong trends, this strategy captures sustained moves while minimizing whipsaws. Volume confirmation ensures institutional participation. Target: 20-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    
    # Proper ER calculation over 10 periods
    er_period = 10
    change = np.abs(np.diff(close, prepend=close[0]))
    # Sum of absolute changes over er_period for numerator
    change_sum = np.zeros(n)
    for i in range(er_period, n):
        change_sum[i] = np.sum(change[i-er_period+1:i+1])
    
    # Sum of absolute price changes over er_period for denominator (volatility)
    volatility_sum = np.zeros(n)
    for i in range(er_period, n):
        volatility_sum[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1], prepend=close[i-er_period])))
    
    # Avoid division by zero
    er = np.zeros(n)
    er[volatility_sum > 0] = change_sum[volatility_sum > 0] / volatility_sum[volatility_sum > 0]
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # for EMA(2)
    slow_sc = 2 / (30 + 1)  # for EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 200-period SMA for trend filter (avoid counter-trend in strong markets)
    sma200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 200  # need 200 for SMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(sma200[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA and above SMA200 with volume spike
            if (close[i] > kama[i] and close[i] > sma200[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA and below SMA200 with volume spike
            elif (close[i] < kama[i] and close[i] < sma200[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Trend_With_200MA_Filter"
timeframe = "4h"
leverage = 1.0