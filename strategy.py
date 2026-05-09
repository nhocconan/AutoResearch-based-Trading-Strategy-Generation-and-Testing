#!/usr/bin/env python3
# Hypothesis: 1d timeframe with weekly ATR-based volatility filter and daily KAMA trend.
# Uses weekly ATR to filter low volatility regimes (avoids chop) and daily KAMA for trend direction.
# Weekly volatility filter reduces whipsaw in sideways markets, KAMA adapts to trend changes.
# Target: 30-100 total trades over 4 years (7-25/year) with size 0.25.

name = "1d_KAMA_Trend_WeeklyATR_Filter"
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
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate daily KAMA trend
    # Efficiency ratio: price change over 10 periods / sum of absolute changes
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[t] - close[t-1]| over window
    # Handle the first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Get weekly ATR for volatility filter (avoid low volatility/chop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1w ATR(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # first value has no TR
    
    atr_14 = np.full_like(tr, np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr_14[i] = np.nanmean(tr[1:15])  # first ATR
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    atr_14_aligned = align_htf_to_ltf(prices, df_1w, atr_14)
    
    # Volatility filter: only trade when weekly ATR > 20-period average ATR
    avg_atr = np.full_like(atr_14_aligned, np.nan)
    for i in range(20, len(atr_14_aligned)):
        if not np.isnan(atr_14_aligned[i-20:i]).all():
            avg_atr[i] = np.nanmean(atr_14_aligned[i-20:i])
    
    volatility_filter = atr_14_aligned > avg_atr
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(volatility_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA + volatility filter
            if close[i] > kama[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA + volatility filter
            elif close[i] < kama[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below KAMA or volatility dies
            if close[i] <= kama[i] or not volatility_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA or volatility dies
            if close[i] >= kama[i] or not volatility_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals