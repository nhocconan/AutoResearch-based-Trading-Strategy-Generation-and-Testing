#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_1D_RSI_Filter
Hypothesis: On 12h timeframe, use Kaufman's Adaptive Moving Average (KAMA) to capture trend direction, filtered by 1D RSI to avoid counter-trend extremes. Long when KAMA > close and RSI < 60; short when KAMA < close and RSI > 40. Exit on opposite KAMA cross. This adapts to volatility, reducing whipsaws in 2022 crash and avoiding overextended entries. Targets 15-25 trades/year with position size 0.25, suitable for 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Calculate KAMA (10,2,30) on 12h data
    kama = np.full(n, np.nan)
    if n >= 30:
        # Efficiency ratio
        change = np.abs(np.diff(close, n=9))  # 10-period change
        volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
        er = np.zeros(n)
        er[9:] = change[9:] / np.where(volatility[9:] != 0, volatility[9:], 1)
        # Smoothing constants
        sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
        # Initialize KAMA
        kama[9] = close[9]
        for i in range(10, n):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get 1D data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on 1D
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    
    for i in range(14, len(close_1d)):
        if i == 14:
            avg_gain[i] = np.mean(gain[0:14])
            avg_loss[i] = np.mean(loss[0:14])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align RSI to 12h timeframe (wait for bar close)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need KAMA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or 
            np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: KAMA > close and RSI not overbought (<60)
            if (kama[i] > close[i] and rsi_1d_aligned[i] < 60):
                signals[i] = 0.25
                position = 1
            # Short entry: KAMA < close and RSI not oversold (>40)
            elif (kama[i] < close[i] and rsi_1d_aligned[i] > 40):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: KAMA crosses below close
            if kama[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA crosses above close
            if kama[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Trend_With_1D_RSI_Filter"
timeframe = "12h"
leverage = 1.0