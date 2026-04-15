#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA + RSI + Chop Filter
# Uses KAMA to determine trend direction, RSI for momentum confirmation,
# and Choppiness Index to filter for trending markets. KAMA adapts to market noise,
# reducing whipsaw in choppy conditions. RSI confirms momentum strength.
# Works in both bull and bear markets by following the KAMA trend.
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (adaptive moving average)
    # Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, 10))
    change[0:10] = 0  # First 10 values
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # Will fix below
    # Recalculate volatility properly
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    # For first 10 periods, use available data
    for i in range(len(close)):
        if i < 10:
            volatility[i] = np.sum(np.abs(np.diff(close[0:i+1]))) if i > 0 else 0
        else:
            volatility[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI (14-period)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (14-period) on 1d
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR (14)
    atr_1d = np.zeros_like(high_1d)
    atr_1d[13] = np.mean(tr[1:14])
    for i in range(14, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Max/Min close over 14 periods
    max_close = np.zeros_like(close_1d)
    min_close = np.zeros_like(close_1d)
    for i in range(13, len(close_1d)):
        max_close[i] = np.max(close_1d[i-13:i+1])
        min_close[i] = np.min(close_1d[i-13:i+1])
    
    # Chop calculation
    sum_tr = np.zeros_like(atr_1d)
    for i in range(13, len(atr_1d)):
        sum_tr[i] = np.sum(tr[i-13:i+1])
    
    chop = np.zeros_like(close_1d)
    for i in range(13, len(close_1d)):
        if max_close[i] != min_close[i]:
            chop[i] = 100 * np.log10(sum_tr[i] / (max_close[i] - min_close[i])) / np.log10(14)
        else:
            chop[i] = 50  # Neutral when no range
    
    # Align KAMA, RSI, and Chop to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            continue
        
        # Long entry: price above KAMA + RSI > 50 (bullish momentum) + Chop < 61.8 (trending)
        if (close[i] > kama_aligned[i] and
            rsi_aligned[i] > 50 and
            chop_aligned[i] < 61.8 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price below KAMA + RSI < 50 (bearish momentum) + Chop < 61.8 (trending)
        elif (close[i] < kama_aligned[i] and
              rsi_aligned[i] < 50 and
              chop_aligned[i] < 61.8 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite KAMA cross or Chop > 61.8 (choppy market)
        elif position == 1 and (close[i] < kama_aligned[i] or chop_aligned[i] > 61.8):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > kama_aligned[i] or chop_aligned[i] > 61.8):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_KAMA_RSI_Chop_Filter"
timeframe = "4h"
leverage = 1.0