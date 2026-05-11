#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI_Pullback_v1
Hypothesis: In trending markets (price > KAMA for long, price < KAMA for short), buy pullbacks to RSI(40) or sell rallies to RSI(60). KAMA adapts to market noise, reducing whipsaws in sideways markets. RSI thresholds avoid extreme overbought/oversold, capturing momentum continuations. Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend). Uses 4h timeframe with minimal conditions to limit trades (<50/year) and reduce fee drag.
"""

name = "4h_KAMA_Trend_RSI_Pullback_v1"
timeframe = "4h"
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
    
    # Efficiency Ratio for KAMA (10-period)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Calculate ER using expanding window for efficiency
    er = np.zeros(n)
    for i in range(n):
        if i < 10:
            er[i] = 0
        else:
            dir_move = np.abs(close[i] - close[i-10])
            vol_sum = np.sum(np.abs(np.diff(close[i-9:i+1])))
            er[i] = dir_move / vol_sum if vol_sum != 0 else 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) with proper initialization
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(n):
        if i < 14:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * i + gain[i]) / (i + 1)
                avg_loss[i] = (avg_loss[i-1] * i + loss[i]) / (i + 1)
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average (20-period)
    vol_ma = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # After warmup for KAMA/RSI
    
    for i in range(start_idx, n):
        # Volume confirmation: avoid low-volume noise
        if vol_ratio[i] < 0.5:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > KAMA (uptrend) + RSI pullback to 40-50 + volume
            if (close[i] > kama[i] and 
                40 <= rsi[i] <= 50 and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA (downtrend) + RSI rally to 50-60 + volume
            elif (close[i] < kama[i] and 
                  50 <= rsi[i] <= 60 and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: trend reversal (price crosses KAMA) or RSI extreme
            if position == 1:
                if close[i] < kama[i] or rsi[i] >= 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if close[i] > kama[i] or rsi[i] <= 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals