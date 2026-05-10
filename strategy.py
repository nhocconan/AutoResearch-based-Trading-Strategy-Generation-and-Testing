#!/usr/bin/env python3
# 4h_KAMA_Trend_With_Volume_Filter_v3
# Hypothesis: KAMA adapts to market noise, providing a reliable trend filter. 
# Price crossing above/below KAMA with volume confirmation (2x MA) captures momentum shifts.
# Works in bull markets by catching uptrends early and in bear markets by following downtrends.
# Uses 4h timeframe with 1d KAMA trend filter to reduce whipsaw and trade frequency.

name = "4h_KAMA_Trend_With_Volume_Filter_v3"
timeframe = "4h"
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
    
    # Get daily data for KAMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily KAMA (ER=10, fast=2, slow=30)
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will compute properly below
    # Recompute volatility correctly: sum of absolute changes over 10 periods
    volatility = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 10:
            volatility[i] = np.nan
        else:
            volatility[i] = np.sum(np.abs(np.diff(close_1d[i-10:i+1])))
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_1d = kama
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Volume confirmation (20-period MA on 4h = ~3.3 days)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (10) and volume MA (20)
    start_idx = max(10, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # KAMA trend filter
        above_kama = close[i] > kama_1d_aligned[i]
        below_kama = close[i] < kama_1d_aligned[i]
        
        # Volume confirmation (stricter: >2.0x MA to reduce false signals)
        volume_confirm = volume[i] > volume_ma[i] * 2.0
        
        if position == 0:
            # Long entry: price above KAMA + volume confirmation
            if above_kama and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price below KAMA + volume confirmation
            elif below_kama and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA
            if below_kama:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA
            if above_kama:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals