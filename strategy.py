#!/usr/bin/env python3
"""
6h_KAMA_Rebound_With_1dTrend_Filter
Hypothesis: Price retracing to 6h KAMA (adaptive moving average) during strong daily trends offers
high-probability bounce entries. KAMA adapts to volatility, reducing whipsaws in chop. 
Daily trend filter ensures alignment with higher timeframe momentum, improving win rate in both bull and bear.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 6h KAMA (adaptive moving average)
    # Efficiency Ratio: |price change| / sum of absolute changes
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    er = np.zeros(n)
    for i in range(10, n):  # ER needs lookback
        er[i] = np.abs(close[i] - close[i-10]) / (np.sum(abs_change[i-9:i+1]) + 1e-10)
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1-day EMA trend filter (34-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_6h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Optional: volatility filter - avoid extremely low volatility
    atr = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup for KAMA and ATR
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(ema_1d_6h[i]) or 
            np.isnan(atr[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        ema_trend = ema_1d_6h[i]
        atr_val = atr[i]
        
        # Entry conditions: price near KAMA with daily trend alignment
        dist_to_kama = np.abs(price - kama_val)
        
        if position == 0:
            # Long: price above KAMA in uptrend, bouncing from below
            if (price > kama_val and 
                ema_trend > kama_val and  # daily trend above KAMA = uptrend
                price < kama_val + 0.5 * atr_val and  # within 0.5 ATR above KAMA
                close[i-1] < kama_val):  # was below KAMA previous bar
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA in downtrend, bouncing from above
            elif (price < kama_val and 
                  ema_trend < kama_val and  # daily trend below KAMA = downtrend
                  price > kama_val - 0.5 * atr_val and  # within 0.5 ATR below KAMA
                  close[i-1] > kama_val):  # was above KAMA previous bar
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below KAMA or trend turns down
            if price < kama_val or ema_trend < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA or trend turns up
            if price > kama_val or ema_trend > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_KAMA_Rebound_With_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0