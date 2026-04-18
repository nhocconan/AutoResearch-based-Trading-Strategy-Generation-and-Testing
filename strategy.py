#!/usr/bin/env python3
"""
12h_KAMA_RSI_200MA_TrendFilter
Hypothesis: KAMA (Kaufman Adaptive MA) adapts to market noise, reducing whipsaw in sideways markets. Combined with RSI momentum filter and 200MA trend filter on 1d timeframe, this should capture strong trends while avoiding false signals in chop. Designed for 12h timeframe to limit trade frequency (target: 12-37 trades/year) and reduce fee drag. Works in both bull (trend following) and bear (avoids false reversals via trend filter).
"""

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
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (200MA) and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for additional trend confirmation (optional, but helps in strong trends)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate KAMA on 12h prices (using close)
    # Efficiency ratio: ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.subtract(close[10:], close[:-10]))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.subtract(close[1:], close[:-1])), axis=0)  # sum of absolute changes
    # Fix: volatility needs to be computed over rolling window
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
        if i >= 10:
            volatility[i] -= np.abs(close[i-10] - close[i-11]) if i-11 >= 0 else 0
    # Avoid division by zero
    er = np.zeros_like(close)
    for i in range(10, len(close)):
        if volatility[i] > 0:
            er[i] = change[i-10] / volatility[i]
        else:
            er[i] = 0
    # Smoothing constants: fastest SC = 2/(2+1)=0.67, slowest SC = 2/(30+1)=0.0645
    sc = (er * 0.6055 + 0.0645) ** 2  # where 0.6055 = (0.67-0.0645)
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    if len(close) > 10:
        kama[10] = close[10]  # seed
        for i in range(11, len(close)):
            if not np.isnan(sc[i]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
    
    # Calculate 1d 200-day SMA for trend filter
    close_1d = df_1d['close'].values
    sma_200_1d = np.full_like(close_1d, np.nan)
    for i in range(199, len(close_1d)):
        sma_200_1d[i] = np.mean(close_1d[i-199:i+1])
    sma_200_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    # Calculate 1d RSI (14-period)
    delta = np.subtract(close_1d[1:], close_1d[:-1])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    for i in range(14, len(close_1d)):
        if i == 14:
            avg_gain[i] = np.mean(gain[:14])
            avg_loss[i] = np.mean(loss[:14])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Optional: 1w trend filter (if available)
    if len(df_1w) >= 50:
        close_1w = df_1w['close'].values
        sma_50_1w = np.full_like(close_1w, np.nan)
        for i in range(49, len(close_1w)):
            sma_50_1w[i] = np.mean(close_1w[i-49:i+1])
        sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    else:
        sma_50_1w_aligned = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after KAMA seed and indicators are ready
    start_idx = max(20, 10)  # KAMA needs ~10 bars, plus buffer
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(sma_200_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price above KAMA (momentum), above 1d 200MA (trend), RSI not overbought
        if close[i] > kama[i] and close[i] > sma_200_1d_aligned[i] and rsi_1d_aligned[i] < 70:
            # Additional 1w trend filter if available
            if len(df_1w) >= 50:
                if close[i] > sma_50_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
                    position = 0
            else:
                signals[i] = 0.25
                position = 1
        # Short conditions: price below KAMA, below 1d 200MA, RSI not oversold
        elif close[i] < kama[i] and close[i] < sma_200_1d_aligned[i] and rsi_1d_aligned[i] > 30:
            if len(df_1w) >= 50:
                if close[i] < sma_50_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
                    position = 0
            else:
                signals[i] = -0.25
                position = -1
        # Exit conditions: price crosses KAMA in opposite direction OR trend fails
        elif position == 1:
            if close[i] < kama[i] or close[i] < sma_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if close[i] > kama[i] or close[i] > sma_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_KAMA_RSI_200MA_TrendFilter"
timeframe = "12h"
leverage = 1.0