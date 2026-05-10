#!/usr/bin/env python3
# 6h_WoodyCCI_ZeroCross_With_1dTrend
# Hypothesis: Woody CCI (commodity channel index) crossing zero with 1d trend filter
# provides timely entries in trending markets while avoiding range-bound whipsaws.
# Uses CCI(14) crossing zero as momentum signal, confirmed by 1d EMA trend.
# Target: 15-25 trades/year to minimize fee drag on 6h timeframe.

name = "6h_WoodyCCI_ZeroCross_With_1dTrend"
timeframe = "6h"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Woody CCI (14) on 6h data
    # Typical Price = (High + Low + Close) / 3
    tp = (high + low + close) / 3.0
    # Moving Average of TP
    ma_tp = pd.Series(tp).rolling(window=14, min_periods=14).mean().values
    # Mean Deviation
    mad = pd.Series(tp).rolling(window=14, min_periods=14).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    # CCI = (TP - MA) / (0.015 * MD)
    cci = (tp - ma_tp) / (0.015 * mad)
    # Replace inf/NaN from zero MD with 0
    cci = np.where(np.isfinite(cci), cci, 0.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need CCI(14), EMA50(1d)
    start_idx = max(14, 50)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(cci[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 1d trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: CCI crosses above zero + uptrend
            if cci[i-1] <= 0 and cci[i] > 0 and uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: CCI crosses below zero + downtrend
            elif cci[i-1] >= 0 and cci[i] < 0 and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: CCI crosses below zero or trend breaks
            if cci[i] < 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: CCI crosses above zero or trend breaks
            if cci[i] > 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals