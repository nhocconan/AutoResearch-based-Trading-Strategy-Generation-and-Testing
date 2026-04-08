#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Filter
Hypothesis: KAMA adapts to market noise, capturing trend direction while avoiding whipsaws. RSI filters overextension, and Choppiness Index identifies ranging vs trending regimes. Works in bull/bear by only taking trend-following entries when market is trending (CHOP < 38.2) and avoiding counter-trend signals in chop. Targets 10-25 trades/year on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA (adaptive moving average) - trend direction
    # Efficiency Ratio: abs(net change over 10 periods) / sum of absolute changes
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # sum of |close[t] - close[t-1]|
    # Fix array alignment: volatility needs to be same length as change
    volatility_full = np.sum(np.abs(np.diff(close)), axis=0)  # This is wrong approach
    
    # Correct ER calculation
    er = np.zeros_like(close)
    for i in range(10, n):
        price_change = np.abs(close[i] - close[i-10])
        sum_abs_change = np.sum(np.abs(np.diff(close[i-9:i+1])))  # sum of 10 absolute changes
        if sum_abs_change > 0:
            er[i] = price_change / sum_abs_change
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) - momentum filter
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Wilder's smoothing
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14) - regime filter
    # True Range
    tr1 = high - low
    tr2 = np.abs(np.concatenate([[high[0]], high[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(np.concatenate([[low[0]], low[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over 14 periods
    atr_sum = np.zeros_like(close)
    for i in range(13, n):
        atr_sum[i] = np.sum(tr[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    highest_high = np.zeros_like(close)
    lowest_low = np.zeros_like(close)
    for i in range(13, n):
        highest_high[i] = np.max(high[i-13:i+1])
        lowest_low[i] = np.min(low[i-13:i+1])
    
    # Chop calculation
    chop = np.zeros_like(close)
    for i in range(13, n):
        if atr_sum[i] > 0 and (highest_high[i] - lowest_low[i]) > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    signals = np.zeros(n)
    
    for i in range(30, n):
        # Skip if any required data is NaN or not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            i < 30):  # ensure we have enough data for all indicators
            continue
            
        # Only trade in trending markets (CHOP < 38.2)
        if chop[i] >= 38.2:
            signals[i] = 0.0
            continue
            
        # Long: price above KAMA AND RSI not overbought (< 70)
        if close[i] > kama[i] and rsi[i] < 70:
            signals[i] = 0.25
        # Short: price below KAMA AND RSI not oversold (> 30)
        elif close[i] < kama[i] and rsi[i] > 30:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals