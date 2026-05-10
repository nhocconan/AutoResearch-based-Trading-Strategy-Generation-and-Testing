#!/usr/bin/env python3
# 12h_KAMA_Trend_With_Volume_Confirmation
# Hypothesis: KAMA adapts to market noise, providing a reliable trend filter.
# In trending markets, price above KAMA indicates uptrend, below indicates downtrend.
# We use 1d trend filter (KAMA) to avoid counter-trend trades and volume confirmation
# to avoid false breakouts. This strategy works in both bull and bear markets by
# only trading in the direction of the 1d KAMA trend. Target: 15-35 trades/year.

name = "12h_KAMA_Trend_With_Volume_Confirmation"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 1d close
    # Efficiency Ratio (ER) over 10 periods
    change = abs(df_1d['close'] - df_1d['close'].shift(10))
    volatility = abs(df_1d['close'].diff()).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, 1e-10)  # avoid division by zero
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Calculate KAMA
    kama = df_1d['close'].copy()
    for i in range(1, len(kama)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (df_1d['close'].iloc[i] - kama.iloc[i-1])
    kama_values = kama.values
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_values)
    
    # Volume confirmation (20-period MA on 12h = ~10 days)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (30), volume MA (20)
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 1d trend filter
        uptrend = close[i] > kama_aligned[i]
        downtrend = close[i] < kama_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: uptrend + volume
            if uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + volume
            elif downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks
            if not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks
            if not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals