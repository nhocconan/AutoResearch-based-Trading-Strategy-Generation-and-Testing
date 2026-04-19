#!/usr/bin/env python3
"""
6h_1d_KAMA_Trend_RSI_Entry
Hypothesis: 6h KAMA identifies the trend direction, 1d RSI filters for overbought/oversold conditions,
and price crosses above/below 6h KAMA with volume confirmation trigger entries.
Designed to work in both bull and bear markets by using trend-following with mean-reversion entries.
Target: 20-50 trades per year to minimize fee drag.
"""

name = "6h_1d_KAMA_Trend_RSI_Entry"
timeframe = "6h"
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
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) parameters
    er_period = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio and smoothing constant
    change = np.abs(np.diff(close, k=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.zeros_like(close)
    er[er_period:] = change[er_period:] / np.maximum(volatility[er_period:], 1e-10)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(kama[i-1]) and not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate RSI(14) on 1d data
    rsi_period = 14
    delta = np.diff(df_1d['close'].values)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(df_1d['close'].values, np.nan)
    avg_loss = np.full_like(df_1d['close'].values, np.nan)
    
    # First average
    if len(gain) >= rsi_period:
        avg_gain[rsi_period-1] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period-1] = np.mean(loss[:rsi_period])
        
        # Wilder smoothing
        for i in range(rsi_period, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above KAMA, RSI not overbought (<70), volume confirmation
            if (close[i] > kama[i] and close[i-1] <= kama[i-1] and
                rsi_1d_aligned[i] < 70 and
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA, RSI not oversold (>30), volume confirmation
            elif (close[i] < kama[i] and close[i-1] >= kama[i-1] and
                  rsi_1d_aligned[i] > 30 and
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below KAMA or RSI overbought
            if (close[i] < kama[i] and close[i-1] >= kama[i-1]) or (rsi_1d_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above KAMA or RSI oversold
            if (close[i] > kama[i] and close[i-1] <= kama[i-1]) or (rsi_1d_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals