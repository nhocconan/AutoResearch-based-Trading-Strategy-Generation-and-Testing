# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
12h_KAMA_Direction_RSI14_ChopFilter
Hypothesis: Use KAMA direction from 1d timeframe to determine trend, enter long when price crosses above KAMA in uptrend, short when price crosses below KAMA in downtrend. Filter entries with RSI(14) > 50 for longs and < 50 for shorts to avoid counter-trend entries. Use Choppiness Index (14) from 1d to avoid ranging markets (CHOP > 61.8). This combines trend-following with momentum and regime filtering to work in both bull and bear markets. Target: 15-25 trades/year on 12h timeframe.
"""

name = "12h_KAMA_Direction_RSI14_ChopFilter"
timeframe = "12h"
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
    
    # Get 1d data for KAMA, RSI, and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average ) on 1d
    # ER = Efficiency Ratio, SC = Smoothing Constant
    def kama(close, length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
        # For array calculation, we need to compute per element
        er = np.zeros_like(close)
        for i in range(length, len(close)):
            if np.sum(np.abs(np.diff(close[i-length:i+1]))) > 0:
                er[i] = np.abs(close[i] - close[i-length]) / np.sum(np.abs(np.diff(close[i-length:i+1])))
            else:
                er[i] = 0
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama_out = np.zeros_like(close)
        kama_out[0] = close[0]
        for i in range(1, len(close)):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        return kama_out
    
    kama_1d = kama(close_1d, length=10, fast=2, slow=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate RSI(14) on 1d
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        for i in range(1, len(close)):
            if i < length:
                avg_gain[i] = (avg_gain[i-1] * (i-1) + gain[i]) / i if i > 0 else gain[i]
                avg_loss[i] = (avg_loss[i-1] * (i-1) + loss[i]) / i if i > 0 else loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
                avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_out = 100 - (100 / (1 + rs))
        return rsi_out
    
    rsi_1d = rsi(close_1d, length=14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate Choppiness Index (14) on 1d
    def chop(high, low, close, length=14):
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            atr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        # True Range calculation
        tr = np.zeros_like(close)
        for i in range(len(close)):
            if i == 0:
                tr[i] = high[i] - low[i]
            else:
                tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        # Sum of TR over period
        tr_sum = np.zeros_like(close)
        for i in range(length-1, len(close)):
            tr_sum[i] = np.sum(tr[i-length+1:i+1])
        # Highest high and lowest low over period
        hh = np.zeros_like(close)
        ll = np.zeros_like(close)
        for i in range(length-1, len(close)):
            hh[i] = np.max(high[i-length+1:i+1])
            ll[i] = np.min(low[i-length+1:i+1])
        # Chop calculation
        chop_out = np.zeros_like(close)
        for i in range(length-1, len(close)):
            if tr_sum[i] > 0 and (hh[i] - ll[i]) > 0:
                chop_out[i] = 100 * np.log10(tr_sum[i] / (hh[i] - ll[i])) / np.log10(length)
            else:
                chop_out[i] = 50  # default to neutral
        return chop_out
    
    chop_1d = chop(high_1d, low_1d, close_1d, length=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chop filter: avoid ranging markets (CHOP > 61.8)
        if chop_1d_aligned[i] > 61.8:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price crosses above KAMA, RSI > 50 (bullish momentum)
            if close[i] > kama_1d_aligned[i] and close[i-1] <= kama_1d_aligned[i-1] and rsi_1d_aligned[i] > 50:
                signals[i] = 0.25
                position = 1
            # Short: Price crosses below KAMA, RSI < 50 (bearish momentum)
            elif close[i] < kama_1d_aligned[i] and close[i-1] >= kama_1d_aligned[i-1] and rsi_1d_aligned[i] < 50:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price crosses below KAMA (trend change)
            if close[i] < kama_1d_aligned[i] and close[i-1] >= kama_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price crosses above KAMA (trend change)
            if close[i] > kama_1d_aligned[i] and close[i-1] <= kama_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals