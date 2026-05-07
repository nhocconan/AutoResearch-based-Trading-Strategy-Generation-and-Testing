#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_Volume_Filter
Hypothesis: Use KAMA (Kaufman Adaptive Moving Average) on 12h for trend direction, filtered by volume spikes and 1w trend (via EMA50). This adaptive trend filter reduces whipsaw in sideways markets while capturing strong trends. Entry when price crosses KAMA with volume confirmation, exit on reverse cross. Designed for 10-30 trades/year on 12h timeframe to minimize fee drag while capturing major moves in both bull and bear markets.
"""
name = "12h_KAMA_Trend_With_Volume_Filter"
timeframe = "12h"
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
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate KAMA on 12h price
    # KAMA parameters: ER with fast=2, slow=30
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close)).rolling(window=10, min_periods=1).sum()
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # smooth constant
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume filter: current volume > 2.0 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if position == 0:
            # Long: price crosses above KAMA + 1w uptrend + volume filter
            if (close[i] > kama[i] and 
                close[i-1] <= kama[i-1] and 
                ema_50_1w_aligned[i] > 0 and  # 1w EMA50 trending up (simplified)
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA + 1w downtrend + volume filter
            elif (close[i] < kama[i] and 
                  close[i-1] >= kama[i-1] and 
                  ema_50_1w_aligned[i] < 0 and  # 1w EMA50 trending down
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA
            if close[i] < kama[i] and close[i-1] >= kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above KAMA
            if close[i] > kama[i] and close[i-1] <= kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals