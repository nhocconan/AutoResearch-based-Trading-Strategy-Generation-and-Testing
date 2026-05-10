#!/usr/bin/env python3
# 4h_KAMA_Trend_1dRSI_34
# Hypothesis: KAMA (Kaufman Adaptive Moving Average) on 4h provides trend direction with low lag.
# Daily RSI(34) acts as a filter: long only when RSI > 50, short only when RSI < 50.
# Volume > 1.5x 20-period MA confirms momentum. This combination reduces whipsaw in both bull and bear markets.
# Target: 15-40 trades/year on 4h, avoiding excessive trade frequency.

name = "4h_KAMA_Trend_1dRSI_34"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 4h close
    # Parameters: ER fast=2, slow=30
    change = np.abs(np.diff(close, prepend=close[0]))
    vol = np.sum(np.abs(np.diff(close, prepend=close[0]))[:1])  # placeholder, will compute properly
    # Proper ER calculation
    dir = np.abs(np.subtract(close[9:], close[:-9]))  # 10-period change
    vol_sum = np.array([np.sum(np.abs(np.diff(close[i:i+10])) for i in range(len(close)-9))])
    # Simplified approach: use standard KAMA calculation
    fast_end = 0.6667  # 2/(2+1)
    slow_end = 0.0645  # 2/(30+1)
    er = np.zeros_like(close)
    ssc = np.zeros_like(close)
    kama = np.zeros_like(close)
    kama[0] = close[0]
    
    # Calculate efficiency ratio and smoothing constant
    for i in range(10, n):
        if i >= 10:
            direction = np.abs(close[i] - close[i-9])
            volatility = np.sum(np.abs(np.diff(close[i-9:i+1])))
            if volatility > 0:
                er[i] = direction / volatility
            else:
                er[i] = 0
            sc = (er[i] * (fast_end - slow_end) + slow_end) ** 2
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
        else:
            kama[i] = close[i]
    
    # Calculate daily RSI(34)
    delta = np.diff(df_1d['close'].values, prepend=df_1d['close'].values[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/34, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/34, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan] * 34, rsi[34:]])  # align with df_1d index
    
    # Align KAMA and daily RSI to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, prices, kama)  # same index
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate volume average for confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Warmup for RSI and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily RSI filter: long when RSI > 50, short when RSI < 50
        rsi_long = rsi_aligned[i] > 50
        rsi_short = rsi_aligned[i] < 50
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: price above KAMA with RSI > 50 and volume confirmation
            if close[i] > kama_aligned[i] and rsi_long and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price below KAMA with RSI < 50 and volume confirmation
            elif close[i] < kama_aligned[i] and rsi_short and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls back below KAMA or RSI drops below 50
            if close[i] < kama_aligned[i] or not rsi_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises back above KAMA or RSI rises above 50
            if close[i] > kama_aligned[i] or not rsi_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals