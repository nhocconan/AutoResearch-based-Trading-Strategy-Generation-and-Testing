#!/usr/bin/env python3
# Hypothesis: 1d KAMA direction + RSI + Chop regime filter
# Long when KAMA trending up, RSI < 70 (avoid overbought), Chop > 61.8 (range)
# Short when KAMA trending down, RSI > 30 (avoid oversold), Chop > 61.8 (range)
# Exit when KAMA reverses direction or Chop < 38.2 (trending)
# Uses Kaufman Adaptive Moving Average for trend, RSI for momentum filter, Choppiness Index for regime
# Designed to capture mean-reversion in ranging markets while avoiding strong trends
# Target: 30-100 total trades over 4 years (7-25/year) with size 0.25

name = "1d_KAMA_RSI_Chop_Range"
timeframe = "1d"
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
    
    # Calculate 1d KAMA (Kaufman Adaptive Moving Average)
    def kama(close, length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.full_like(close, np.nan, dtype=float)
        kama[length] = close[length]
        for i in range(length+1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Calculate 1d RSI (Relative Strength Index)
    def rsi(close, length=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[length] = np.mean(gain[:length])
        avg_loss[length] = np.mean(loss[:length])
        for i in range(length+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Calculate 1d Choppiness Index
    def choppiness_index(high, low, close, length=14):
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            atr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        sum_atr = np.zeros_like(close)
        for i in range(length, len(close)):
            sum_atr[i] = np.sum(atr[i-length+1:i+1])
        max_range = np.zeros_like(close)
        for i in range(length-1, len(close)):
            max_range[i] = np.max(high[i-length+1:i+1]) - np.min(low[i-length+1:i+1])
        cpi = np.where(max_range != 0, 100 * np.log10(sum_atr / max_range) / np.log10(length), 50)
        return cpi
    
    # Load 1d data (already the timeframe we're working with)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate indicators on 1d data
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    kama_1d = kama(close_1d, length=10, fast=2, slow=30)
    rsi_1d = rsi(close_1d, length=14)
    chop_1d = choppiness_index(high_1d, low_1d, close_1d, length=14)
    
    # Align indicators to 1d timeframe (no alignment needed since same timeframe)
    kama_aligned = kama_1d
    rsi_aligned = rsi_1d
    chop_aligned = chop_1d
    
    # Load 1h data for trend confirmation (optional)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) >= 10:
        # Simple trend: price above/below 10-period SMA on 1h
        sma_10_1h = pd.Series(df_1h['close']).rolling(window=10, min_periods=10).mean().values
        sma_10_aligned = align_htf_to_ltf(prices, df_1h, sma_10_1h)
    else:
        sma_10_aligned = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA up, RSI not overbought, choppy market
            if (kama_aligned[i] > kama_aligned[i-1] and 
                rsi_aligned[i] < 70 and 
                chop_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA down, RSI not oversold, choppy market
            elif (kama_aligned[i] < kama_aligned[i-1] and 
                  rsi_aligned[i] > 30 and 
                  chop_aligned[i] > 61.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA reverses down or market trends
            if (kama_aligned[i] < kama_aligned[i-1]) or (chop_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA reverses up or market trends
            if (kama_aligned[i] > kama_aligned[i-1]) or (chop_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals