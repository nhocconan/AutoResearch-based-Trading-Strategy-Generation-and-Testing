#!/usr/bin/env python3
"""
1d_weekly_rsi_mean_reversion
Hypothesis: On daily timeframe, buy when weekly RSI < 30 and sell when weekly RSI > 70, using 1-week timeframe for RSI calculation. This captures mean reversion in longer-term momentum extremes while avoiding short-term noise. Works in both bull and bear markets as RSI extremes tend to reverse regardless of trend. Targets 10-20 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_rsi_mean_reversion"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    
    # Calculate weekly RSI from 1w timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate RSI on weekly close
    weekly_close = df_1w['close'].values
    delta = np.diff(weekly_close, prepend=weekly_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to RMA)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # First 14-period average
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)  # Handle no losses
    
    # Align weekly RSI to daily timeframe
    rsi_1d = align_htf_to_ltf(prices, df_1w, rsi)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        if np.isnan(rsi_1d[i]):
            signals[i] = 0.0
            continue
            
        # Enter long when weekly RSI < 30 (oversold)
        if rsi_1d[i] < 30:
            signals[i] = 0.25
        # Enter short when weekly RSI > 70 (overbought)
        elif rsi_1d[i] > 70:
            signals[i] = -0.25
        # Otherwise flat
        else:
            signals[i] = 0.0
    
    return signals