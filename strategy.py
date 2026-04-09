#!/usr/bin/env python3
# 1d_1w_rsi_mean_reversion_v1
# Hypothesis: On 1d timeframe, use weekly RSI (from 1w timeframe) as trend filter and daily RSI for mean-reversion entries.
# In bull markets (weekly RSI > 50), look for daily RSI < 30 to go long. In bear markets (weekly RSI < 50), look for daily RSI > 70 to go short.
# Weekly RSI acts as regime filter to avoid counter-trend trades. Daily RSI provides mean-reversion signals.
# Target: 10-25 trades per year per symbol (~40-100 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_rsi_mean_reversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate weekly RSI(14)
    weekly_close = df_1w['close'].values
    delta = np.diff(weekly_close, prepend=weekly_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(weekly_close)
    avg_loss = np.zeros_like(weekly_close)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(weekly_close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    weekly_rsi = 100 - (100 / (1 + rs))
    
    # Align weekly RSI to daily timeframe
    weekly_rsi_aligned = align_htf_to_ltf(prices, df_1w, weekly_rsi)
    
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
    daily_rsi = 100 - (100 / (1 + rs_d))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(weekly_rsi_aligned[i]) or np.isnan(daily_rsi[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: daily RSI returns to 50 or weekly RSI flips bearish
            if daily_rsi[i] >= 50 or weekly_rsi_aligned[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: daily RSI returns to 50 or weekly RSI flips bullish
            if daily_rsi[i] <= 50 or weekly_rsi_aligned[i] >= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: bullish weekly regime + oversold daily RSI
            if weekly_rsi_aligned[i] >= 50 and daily_rsi[i] < 30:
                position = 1
                signals[i] = 0.25
            # Enter short: bearish weekly regime + overbought daily RSI
            elif weekly_rsi_aligned[i] < 50 and daily_rsi[i] > 70:
                position = -1
                signals[i] = -0.25
    
    return signals