#!/usr/bin/env python3
"""
6h_KAMA_Trend_With_RSI_Divergence
Hypothesis: Use KAMA (Kaufman Adaptive Moving Average) on 6h for adaptive trend detection, combined with RSI divergence on 1d for early reversal signals. Works in both bull and bear markets by adapting to volatility and using divergences to catch trend changes early.
Target: 15-35 trades/year to minimize fee drag while capturing meaningful trend shifts.
"""

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
    
    # Get 1d data for RSI divergence
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate KAMA on 6h (adaptive trend)
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # needs correction
    # Correct approach: loop for ER calculation
    er = np.zeros(n)
    for i in range(10, n):
        if i >= 10:
            price_change = np.abs(close[i] - close[i-10])
            volatility_sum = np.sum(np.abs(np.diff(close[i-10:i+1])))
            if volatility_sum > 0:
                er[i] = price_change / volatility_sum
            else:
                er[i] = 0
    # Smoothing constants
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI on 1d for divergence
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)
    rsi = np.where(avg_gain == 0, 0, rsi)
    
    # Align KAMA and RSI to 6h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)  # KAMA is already 6h, but we align to ensure sync
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate RSI divergence: bullish when price makes lower low but RSI makes higher low
    # Bearish when price makes higher high but RSI makes lower high
    # We'll look for divergences over 5-period windows
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    
    lookback = 5
    for i in range(lookback, n):
        # Bullish divergence: price lower low, RSI higher low
        if low[i] < low[i-lookback] and rsi_aligned[i] > rsi_aligned[i-lookback]:
            # Check if it's a meaningful divergence
            bullish_div[i] = True
        # Bearish divergence: price higher high, RSI lower high
        if high[i] > high[i-lookback] and rsi_aligned[i] < rsi_aligned[i-lookback]:
            bearish_div[i] = True
    
    # Trend filter: price above/below KAMA
    uptrend = close > kama_aligned
    downtrend = close < kama_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: RSI divergence in direction of trend
        long_entry = bullish_div[i] and uptrend[i]
        short_entry = bearish_div[i] and downtrend[i]
        
        # Exit conditions: opposite divergence or trend change
        long_exit = bearish_div[i] or (not uptrend[i])
        short_exit = bullish_div[i] or (not downtrend[i])
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_KAMA_Trend_With_RSI_Divergence"
timeframe = "6h"
leverage = 1.0