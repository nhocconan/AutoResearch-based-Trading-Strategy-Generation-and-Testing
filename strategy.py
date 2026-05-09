#!/usr/bin/env python3
# 4h_KAMA_Trend_RSI_Extremes_ChopFilter
# Hypothesis: Kaufman Adaptive Moving Average (KAMA) defines trend direction, RSI extremes provide entry signals in trending markets, and Choppiness Index filter avoids ranging conditions. Works in bull/bear: KAMA adapts to volatility, RSI extremes capture mean-reversion within trend, chop filter prevents whipsaws in sideways markets.

name = "4h_KAMA_Trend_RSI_Extremes_ChopFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (10, 2, 30) for trend
    def calculate_kama(close, fast=2, slow=30, length=10):
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.zeros_like(close)
        for i in range(length, len(close)):
            if volatility[i] != 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[length] = close[length]
        for i in range(length+1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, fast=2, slow=30, length=10)
    
    # Calculate RSI (14)
    def calculate_rsi(close, length=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[length] = np.mean(gain[0:length])
        avg_loss[length] = np.mean(loss[0:length])
        for i in range(length+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, length=14)
    
    # Calculate Choppiness Index (14)
    def calculate_chop(high, low, close, length=14):
        atr = np.zeros_like(close)
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr[1:] = tr
        for i in range(length+1, len(close)):
            atr[i] = (atr[i-1] * (length-1) + tr[i]) / length
        
        hh = np.zeros_like(close)
        ll = np.zeros_like(close)
        hh[length] = np.max(high[0:length+1])
        ll[length] = np.min(low[0:length+1])
        for i in range(length+1, len(close)):
            hh[i] = max(hh[i-1], high[i])
            ll[i] = min(ll[i-1], low[i])
        
        sum_atr = np.zeros_like(close)
        for i in range(length, len(close)):
            sum_atr[i] = np.sum(atr[i-length+1:i+1])
        
        chop = 100 * np.log10(sum_atr / (hh - ll)) / np.log10(length)
        return chop
    
    chop = calculate_chop(high, low, close, length=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above KAMA (uptrend) AND RSI < 30 (oversold) AND chop < 61.8 (trending)
            if (close[i] > kama[i] and 
                rsi[i] < 30 and 
                chop[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA (downtrend) AND RSI > 70 (overbought) AND chop < 61.8 (trending)
            elif (close[i] < kama[i] and 
                  rsi[i] > 70 and 
                  chop[i] < 61.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below KAMA OR RSI > 70 (overbought)
            if close[i] < kama[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above KAMA OR RSI < 30 (oversold)
            if close[i] > kama[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals