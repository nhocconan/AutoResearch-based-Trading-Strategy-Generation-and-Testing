#!/usr/bin/env python3
# 6h_1d_cci_momentum_v1
# Hypothesis: 6-hour CCI momentum with 1-day trend filter. Uses CCI(20) > 100 for long momentum and < -100 for short momentum,
# filtered by 1-day EMA(50) trend direction. Works in trending markets (both bull and bear) by aligning with higher timeframe trend.
# Target: 15-35 trades per year per symbol (~60-140 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_cci_momentum_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate CCI(20) on 6h data
    typical_price = (high + low + close) / 3.0
    tp_mean = np.full(n, np.nan)
    tp_dev = np.full(n, np.nan)
    
    for i in range(n):
        if i >= 19:  # 20-period lookback
            tp_slice = typical_price[i-19:i+1]
            tp_mean[i] = np.mean(tp_slice)
            tp_dev[i] = np.mean(np.abs(tp_slice - tp_mean[i]))
    
    cci = np.full(n, np.nan)
    for i in range(19, n):
        if tp_dev[i] > 0:
            cci[i] = (typical_price[i] - tp_mean[i]) / (0.015 * tp_dev[i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(cci[i]) or np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI falls below zero or trend turns bearish
            if cci[i] < 0 or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI rises above zero or trend turns bullish
            if cci[i] > 0 or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: CCI > 100 and price above 1-day EMA(50) (bullish trend)
            if cci[i] > 100 and close[i] > ema_50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: CCI < -100 and price below 1-day EMA(50) (bearish trend)
            elif cci[i] < -100 and close[i] < ema_50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals