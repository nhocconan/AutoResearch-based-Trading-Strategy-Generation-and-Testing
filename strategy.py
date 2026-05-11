#!/usr/bin/env python3
name = "12h_1w_1d_KAMA_RSI_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly and daily data
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly KAMA for trend direction
    def kama(close, length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close)).cumsum()
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_1w = kama(df_1w['close'].values, 10, 2, 30)
    kama_1w_prev = np.roll(kama_1w, 1)
    kama_1w_prev[0] = kama_1w[0]
    trend_up_1w = kama_1w > kama_1w_prev
    trend_down_1w = kama_1w < kama_1w_prev
    
    # Daily RSI(14) for momentum
    delta = np.diff(df_1d['close'], prepend=df_1d['close'][0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align to 12h
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    trend_down_aligned = align_htf_to_ltf(prices, df_1w, trend_down_1w)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]) or
            np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA up + RSI > 50 + volume surge
            if (trend_up_aligned[i] and 
                rsi_1d_aligned[i] > 50 and 
                volume[i] > 1.8 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA down + RSI < 50 + volume surge
            elif (trend_down_aligned[i] and 
                  rsi_1d_aligned[i] < 50 and 
                  volume[i] > 1.8 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA down or RSI < 40
            if (not trend_up_aligned[i] or rsi_1d_aligned[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA up or RSI > 60
            if (not trend_down_aligned[i] or rsi_1d_aligned[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals