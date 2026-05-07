#!/usr/bin/env python3
# 1D_KAMA_1WTrend_Volume_Signal
# Hypothesis: On the daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction, weekly timeframe for long-term trend filter, and volume confirmation (1.5x 20-day average volume) to enter trades. Exit on opposite KAMA cross. Designed for low trade frequency (10-25/year) to avoid fee drag, with trend + volume confluence working in both bull and bear markets by avoiding choppy periods via adaptive trend filter.

name = "1D_KAMA_1WTrend_Volume_Signal"
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
    volume = prices['volume'].values
    
    # Get weekly data for long-term trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly KAMA for trend filter (using close)
    close_1w = df_1w['close'].values
    # ER (Efficiency Ratio) = abs(net change) / sum(abs(changes)) over 10 periods
    change = np.abs(np.diff(close_1w))
    abs_change = np.sum(np.abs(np.diff(close_1w)))
    # Avoid division by zero
    er = np.zeros_like(close_1w)
    for i in range(10, len(close_1w)):
        net_change = abs(close_1w[i] - close_1w[i-10])
        sum_abs = np.sum(np.abs(np.diff(close_1w[i-10:i+1])))
        if sum_abs > 0:
            er[i] = net_change / sum_abs
        else:
            er[i] = 0
    # Smoothing constants: fastest SC = 2/(2+1) = 0.67, slowest SC = 2/(30+1) = 0.0645
    sc = (er * (0.67 - 0.0645) + 0.0645) ** 2
    # Initialize KAMA
    kama_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) > 10:
        kama_1w[10] = close_1w[10]
        for i in range(11, len(close_1w)):
            kama_1w[i] = kama_1w[i-1] + sc[i] * (close_1w[i] - kama_1w[i-1])
    
    # Align weekly KAMA to daily timeframe (use previous week's value)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Daily KAMA for entry signal (same calculation on daily data)
    change_d = np.abs(np.diff(close))
    sc_d = np.zeros_like(close)
    for i in range(10, len(close)):
        net_change_d = abs(close[i] - close[i-10])
        sum_abs_d = np.sum(np.abs(np.diff(close[i-10:i+1])))
        if sum_abs_d > 0:
            er_d = net_change_d / sum_abs_d
        else:
            er_d = 0
        sc_d[i] = (er_d * (0.67 - 0.0645) + 0.0645) ** 2
    kama_d = np.full_like(close, np.nan)
    if len(close) > 10:
        kama_d[10] = close[10]
        for i in range(11, len(close)):
            kama_d[i] = kama_d[i-1] + sc_d[i] * (close[i] - kama_d[i-1])
    
    # Volume confirmation: 1.5x 20-day average volume
    vol_ma = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 11)  # Ensure volume MA and KAMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama_d[i]) or np.isnan(kama_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above daily KAMA (bullish momentum), price above weekly KAMA (long-term uptrend), volume spike
            if (close[i] > kama_d[i] and 
                close[i] > kama_1w_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below daily KAMA (bearish momentum), price below weekly KAMA (long-term downtrend), volume spike
            elif (close[i] < kama_d[i] and 
                  close[i] < kama_1w_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below daily KAMA (trend change)
            if close[i] < kama_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above daily KAMA (trend change)
            if close[i] > kama_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals