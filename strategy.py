#!/usr/bin/env python3
"""
Hypothesis: 4-hour KAMA (Kaufman Adaptive Moving Average) with 1-day RSI filter.
Long when KAMA slope > 0 and 1-day RSI < 55.
Short when KAMA slope < 0 and 1-day RSI > 45.
Exit when KAMA slope reverses or 1-day RSI reaches extremes (30/70).
KAMA adapts to market noise, reducing whipsaw in ranging markets. RSI filter avoids overextended moves.
Designed for 4h timeframe with 1-day HTF to work in both bull and bear markets by following adaptive trend with momentum filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # KAMA parameters
    er_period = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    
    # Proper ER calculation
    er = np.zeros(n)
    for i in range(er_period, n):
        price_change = np.abs(close[i] - close[i - er_period])
        volatility_sum = np.sum(np.abs(np.diff(close[i - er_period + 1:i + 1])))
        if volatility_sum > 0:
            er[i] = price_change / volatility_sum
        else:
            er[i] = 0
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA slope (1-period change)
    kama_slope = np.diff(kama, prepend=0)
    
    # Load 1-day data for RSI filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[:13] = np.nan
    
    # Align RSI to 4h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if np.isnan(kama[i]) or np.isnan(kama_slope[i]) or np.isnan(rsi_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA rising and RSI not overbought
            if kama_slope[i] > 0 and rsi_1d_aligned[i] < 55:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling and RSI not oversold
            elif kama_slope[i] < 0 and rsi_1d_aligned[i] > 45:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: KAMA falls or RSI overbought
                if kama_slope[i] <= 0 or rsi_1d_aligned[i] >= 70:
                    exit_signal = True
            else:  # position == -1
                # Exit short: KAMA rises or RSI oversold
                if kama_slope[i] >= 0 or rsi_1d_aligned[i] <= 30:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_KAMA_1dRSI_Filter"
timeframe = "4h"
leverage = 1.0