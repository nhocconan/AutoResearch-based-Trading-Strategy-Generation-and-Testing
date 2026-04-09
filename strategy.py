#!/usr/bin/env python3
# 1d_1w_rsi_reversal_v1
# Hypothesis: On daily timeframe, buy when weekly RSI is oversold (<30) and daily RSI crosses above 30,
# sell when weekly RSI is overbought (>70) and daily RSI crosses below 70.
# Uses weekly trend filter to avoid counter-trend trades, reducing whipsaw in sideways markets.
# Target: 20-60 trades over 4 years (5-15/year) to minimize fee drag.
# Works in both bull and bear markets as RSI captures reversals and weekly filter ensures trend alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_rsi_reversal_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    rs = np.full(n, np.nan)
    rsi = np.full(n, np.nan)
    
    # Wilder's smoothing
    if n >= 14:
        avg_gain[13] = np.mean(gain[1:14])
        avg_loss[13] = np.mean(loss[1:14])
        for i in range(14, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
            rs[i] = avg_gain[i] / avg_loss[i] if avg_loss[i] != 0 else np.inf
            rsi[i] = 100 - (100 / (1 + rs[i]))
    
    # Load weekly data ONCE before loop for RSI filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly RSI(14)
    close_1w = df_1w['close'].values
    delta_1w = np.diff(close_1w, prepend=close_1w[0])
    gain_1w = np.where(delta_1w > 0, delta_1w, 0)
    loss_1w = np.where(delta_1w < 0, -delta_1w, 0)
    
    avg_gain_1w = np.full(len(df_1w), np.nan)
    avg_loss_1w = np.full(len(df_1w), np.nan)
    rs_1w = np.full(len(df_1w), np.nan)
    rsi_1w = np.full(len(df_1w), np.nan)
    
    if len(df_1w) >= 14:
        avg_gain_1w[13] = np.mean(gain_1w[1:14])
        avg_loss_1w[13] = np.mean(loss_1w[1:14])
        for i in range(14, len(df_1w)):
            avg_gain_1w[i] = (avg_gain_1w[i-1] * 13 + gain_1w[i]) / 14
            avg_loss_1w[i] = (avg_loss_1w[i-1] * 13 + loss_1w[i]) / 14
            rs_1w[i] = avg_gain_1w[i] / avg_loss_1w[i] if avg_loss_1w[i] != 0 else np.inf
            rsi_1w[i] = 100 - (100 / (1 + rs_1w[i]))
    
    # Align weekly RSI to daily timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(rsi[i]) or np.isnan(rsi_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: daily RSI crosses below 50 (momentum fade)
            if rsi[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: daily RSI crosses above 50 (momentum fade)
            if rsi[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: weekly RSI oversold AND daily RSI crosses above 30
            if rsi_1w_aligned[i] < 30 and rsi[i] > 30 and rsi[i-1] <= 30:
                position = 1
                signals[i] = 0.25
            # Enter short: weekly RSI overbought AND daily RSI crosses below 70
            elif rsi_1w_aligned[i] > 70 and rsi[i] < 70 and rsi[i-1] >= 70:
                position = -1
                signals[i] = -0.25
    
    return signals