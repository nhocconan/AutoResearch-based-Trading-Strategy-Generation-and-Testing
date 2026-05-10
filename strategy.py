#!/usr/bin/env python3
# 12h_KAMA_RSI_Chop_Filter
# Hypothesis: 12-hour KAMA direction filtered by RSI momentum and Choppiness regime. 
# KAMA adapts to trend strength, RSI filters momentum extremes, Choppiness avoids whipsaws in sideways markets.
# Designed for low trade frequency (12-37/year) with discipline to work in both bull and bear markets.

name = "12h_KAMA_RSI_Chop_Filter"
timeframe = "12h"
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
    
    # Daily data for KAMA, RSI, and Choppiness
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - ER=10, Fast=2, Slow=30
    def kama(close, er_length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close))
        er = np.zeros_like(close)
        for i in range(1, len(close)):
            if volatility[i-er_length+1:i+1].sum() != 0:
                er[i] = change[i] / volatility[i-er_length+1:i+1].sum()
            else:
                er[i] = 0
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama_out = np.zeros_like(close)
        kama_out[0] = close[0]
        for i in range(1, len(close)):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        return kama_out
    
    # RSI (14)
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        for i in range(1, len(close)):
            if i < length:
                avg_gain[i] = np.mean(gain[1:i+1]) if i > 0 else 0
                avg_loss[i] = np.mean(loss[1:i+1]) if i > 0 else 0
            else:
                avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
                avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi_out = 100 - (100 / (1 + rs))
        return rsi_out
    
    # Choppiness Index (14)
    def choppiness(high, low, close, length=14):
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            atr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        sum_atr = np.zeros_like(close)
        for i in range(length-1, len(close)):
            sum_atr[i] = np.sum(atr[i-length+1:i+1])
        highest = np.zeros_like(close)
        lowest = np.zeros_like(close)
        for i in range(length-1, len(close)):
            highest[i] = np.max(high[i-length+1:i+1])
            lowest[i] = np.min(low[i-length+1:i+1])
        chop = np.zeros_like(close)
        for i in range(length-1, len(close)):
            if highest[i] - lowest[i] != 0:
                chop[i] = 100 * np.log10(sum_atr[i] / (highest[i] - lowest[i])) / np.log10(length)
            else:
                chop[i] = 50
        return chop
    
    # Calculate indicators
    kama_1d = kama(close_1d, er_length=10, fast=2, slow=30)
    rsi_1d = rsi(close_1d, length=14)
    chop_1d = choppiness(high_1d, low_1d, close_1d, length=14)
    
    # Align to 12h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA, RSI > 50 (bullish momentum), Choppiness < 61.8 (trending)
            if close[i] > kama_1d_aligned[i] and rsi_1d_aligned[i] > 50 and chop_1d_aligned[i] < 61.8:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI < 50 (bearish momentum), Choppiness < 61.8 (trending)
            elif close[i] < kama_1d_aligned[i] and rsi_1d_aligned[i] < 50 and chop_1d_aligned[i] < 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below KAMA or RSI < 40 or Choppiness > 61.8 (choppy)
            if close[i] < kama_1d_aligned[i] or rsi_1d_aligned[i] < 40 or chop_1d_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above KAMA or RSI > 60 or Choppiness > 61.8 (choppy)
            if close[i] > kama_1d_aligned[i] or rsi_1d_aligned[i] > 60 or chop_1d_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals