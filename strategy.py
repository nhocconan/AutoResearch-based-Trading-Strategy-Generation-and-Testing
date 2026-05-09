#!/usr/bin/env python3
# Hypothesis: 1d KAMA direction + RSI(14) + Chop regime (14) filter - Long when KAMA rising, RSI < 30, Chop > 61.8; Short when KAMA falling, RSI > 70, Chop > 61.8
# Uses 1d timeframe to reduce trade frequency, with Chop filter to identify ranging markets for mean reversion
# KAMA adapts to market noise, RSI identifies extremes, Chop confirms ranging conditions
# Designed to work in both bull and bear markets by fading extremes in ranging conditions
# Target: 20-60 total trades over 4 years (5-15/year) with size 0.25

name = "1d_KAMA_RSI_Chop_MeanReversion"
timeframe = "1d"
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
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)  # Will fix below
    
    # Proper volatility calculation for ER
    volatility = np.zeros(n)
    for i in range(1, n):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
        if i >= 10:
            volatility[i] -= np.abs(close[i-10] - close[i-11]) if i-11 >= 0 else 0
    
    # Avoid division by zero
    volatility = np.where(volatility == 0, 1e-10, volatility)
    er = change / volatility
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # Fast=2, Slow=30
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index(14)
    atr = np.zeros(n)
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - np.roll(close, 1))
    tr3 = np.abs(np.roll(low, 1) - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    # Calculate ATR
    atr[13] = np.mean(tr[1:14])
    for i in range(14, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = np.zeros(n)
    lowest_low = np.zeros(n)
    for i in range(n):
        if i < 13:
            highest_high[i] = np.max(high[:i+1])
            lowest_low[i] = np.min(low[:i+1])
        else:
            highest_high[i] = np.max(high[i-13:i+1])
            lowest_low[i] = np.min(low[i-13:i+1])
    
    # Avoid division by zero in Chop calculation
    atr_sum = np.zeros(n)
    for i in range(n):
        if i < 13:
            atr_sum[i] = np.sum(tr[:i+1])
        else:
            atr_sum[i] = np.sum(tr[i-13:i+1])
    
    range_14 = highest_high - lowest_low
    chop = np.where(range_14 == 0, 50, 100 * np.log10(atr_sum / range_14) / np.log10(14))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA rising, RSI < 30 (oversold), Chop > 61.8 (ranging)
            if (kama[i] > kama[i-1] and 
                rsi[i] < 30 and 
                chop[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA falling, RSI > 70 (overbought), Chop > 61.8 (ranging)
            elif (kama[i] < kama[i-1] and 
                  rsi[i] > 70 and 
                  chop[i] > 61.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA falling or RSI > 50 (mean reversion complete)
            if (kama[i] < kama[i-1]) or (rsi[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA rising or RSI < 50 (mean reversion complete)
            if (kama[i] > kama[i-1]) or (rsi[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals