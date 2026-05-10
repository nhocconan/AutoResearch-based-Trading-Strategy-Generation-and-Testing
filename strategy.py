#!/usr/bin/env python3
# 4h_KAMA_Trend_Reversal_RSI_Filter
# Hypothesis: KAMA adapts to market noise, providing reliable trend direction with less whipsaw.
# Entry: KAMA direction + RSI extreme + volume confirmation.
# Exit: Opposite KAMA direction or RSI normalization.
# Works in bull markets via trend continuation and in bear via mean reversion at extremes.
# Low trade frequency expected due to triple confirmation.

name = "4h_KAMA_Trend_Reversal_RSI_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.subtract.accumulate(change))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d KAMA for trend filter
    kama_1d = calculate_kama(df_1d['close'].values)
    kama_1d_slope = np.diff(kama_1d, prepend=kama_1d[0])
    kama_1d_up = kama_1d_slope > 0
    
    # Align 1d KAMA trend to 4h
    kama_1d_up_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_up.astype(float))
    
    # Get 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h KAMA for entry signals
    kama_4h = calculate_kama(close)
    kama_4h_slope = np.diff(kama_4h, prepend=kama_4h[0])
    
    # Calculate RSI
    rsi = calculate_rsi(close)
    
    # Volume filter: current volume > 1.3x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (30) + RSI (14) + vol EMA (20)
    start_idx = max(30, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_4h[i]) or 
            np.isnan(kama_1d_up_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1d KAMA up AND 4h KAMA turning up AND RSI < 30 (oversold) AND volume
            if (kama_1d_up_aligned[i] > 0.5 and 
                kama_4h_slope[i] > 0 and 
                rsi[i] < 30 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: 1d KAMA down AND 4h KAMA turning down AND RSI > 70 (overbought) AND volume
            elif (kama_1d_up_aligned[i] <= 0.5 and 
                  kama_4h_slope[i] < 0 and 
                  rsi[i] > 70 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: 1d KAMA down OR 4h KAMA turning down OR RSI > 50
            if (kama_1d_up_aligned[i] <= 0.5 or 
                kama_4h_slope[i] < 0 or 
                rsi[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: 1d KAMA up OR 4h KAMA turning up OR RSI < 50
            if (kama_1d_up_aligned[i] > 0.5 or 
                kama_4h_slope[i] > 0 or 
                rsi[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals