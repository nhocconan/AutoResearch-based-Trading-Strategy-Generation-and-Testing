#!/usr/bin/env python3
# daily_1w_rsi_mean_reversion_v1
# Hypothesis: On daily timeframe, use weekly RSI(14) as trend filter and daily RSI(14) for mean-reversion entries.
# In bull markets (weekly RSI > 50), look for daily RSI < 30 to go long.
# In bear markets (weekly RSI < 50), look for daily RSI > 70 to go short.
# Exit when daily RSI returns to 50 (mean reversion).
# Weekly RSI avoids false signals during strong trends; daily RSI captures short-term overextensions.
# Works in both bull and bear by adapting to weekly trend context.
# Target: 10-25 trades/year with low frequency to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "daily_1w_rsi_mean_reversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get weekly data for RSI trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate weekly RSI(14)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(close_1w)
    avg_loss = np.zeros_like(close_1w)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close_1w)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1w = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
    rsi_1w[:13] = np.nan
    
    # Calculate daily RSI(14)
    delta_d = np.diff(close, prepend=close[0])
    gain_d = np.where(delta_d > 0, delta_d, 0)
    loss_d = np.where(delta_d < 0, -delta_d, 0)
    
    avg_gain_d = np.zeros_like(close)
    avg_loss_d = np.zeros_like(close)
    avg_gain_d[13] = np.mean(gain_d[1:14])
    avg_loss_d[13] = np.mean(loss_d[1:14])
    
    for i in range(14, len(close)):
        avg_gain_d[i] = (avg_gain_d[i-1] * 13 + gain_d[i]) / 14
        avg_loss_d[i] = (avg_loss_d[i-1] * 13 + loss_d[i]) / 14
    
    rs_d = np.where(avg_loss_d != 0, avg_gain_d / avg_loss_d, 100)
    rsi_d = np.where(avg_loss_d == 0, 100, 100 - (100 / (1 + rs_d)))
    rsi_d[:13] = np.nan
    
    # Align weekly RSI to daily timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if data not ready
        if np.isnan(rsi_d[i]) or np.isnan(rsi_1w_aligned[i]):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long
            # Exit: daily RSI returns to 50 (mean reversion)
            if rsi_d[i] >= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: daily RSI returns to 50 (mean reversion)
            if rsi_d[i] <= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: weekly RSI > 50 (bull trend) and daily RSI < 30 (oversold)
            if rsi_1w_aligned[i] > 50 and rsi_d[i] < 30:
                position = 1
                signals[i] = 0.25
            # Enter short: weekly RSI < 50 (bear trend) and daily RSI > 70 (overbought)
            elif rsi_1w_aligned[i] < 50 and rsi_d[i] > 70:
                position = -1
                signals[i] = -0.25
    
    return signals