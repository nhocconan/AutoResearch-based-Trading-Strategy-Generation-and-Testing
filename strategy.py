#!/usr/bin/env python3
"""
1h_HTF_4h1d_DualTrend_Filter
Hypothesis: Uses 4h EMA(34) and 1d EMA(34) as dual trend filters for 1h entries.
Long when both timeframes show uptrend, short when both show downtrend.
Requires 1h price to be near EMA(34) for pullback entries, reducing whipsaw.
Targets 15-30 trades/year by requiring confluence of two higher timeframes.
Works in bull/bear by following established trends on higher timeframes.
"""

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
    
    # Get 4h and 1d data for dual trend filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA(34) for trend filter
    close_4h = df_4h['close'].values
    ema34_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 34:
        ema34_4h[33] = np.mean(close_4h[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_4h)):
            ema34_4h[i] = close_4h[i] * alpha + ema34_4h[i-1] * (1 - alpha)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = close_1d[i] * alpha + ema34_1d[i-1] * (1 - alpha)
    
    # Align HTF EMAs to 1h timeframe
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1h EMA(34) for pullback entry reference
    ema34_1h = np.full(n, np.nan)
    if n >= 34:
        ema34_1h[33] = np.mean(close[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, n):
            ema34_1h[i] = close[i] * alpha + ema34_1h[i-1] * (1 - alpha)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(ema34_1h[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend alignment: both 4h and 1d must agree
        uptrend_4h = close[i] > ema34_4h_aligned[i]
        uptrend_1d = close[i] > ema34_1d_aligned[i]
        downtrend_4h = close[i] < ema34_4h_aligned[i]
        downtrend_1d = close[i] < ema34_1d_aligned[i]
        
        both_uptrend = uptrend_4h and uptrend_1d
        both_downtrend = downtrend_4h and downtrend_1d
        
        # Price near 1h EMA for pullback entry (within 1%)
        near_ema = abs(close[i] - ema34_1h[i]) / ema34_1h[i] < 0.01
        
        if position == 0:
            # Long: both timeframes uptrend + price near 1h EMA (pullback)
            if both_uptrend and near_ema:
                signals[i] = 0.20
                position = 1
            # Short: both timeframes downtrend + price near 1h EMA (pullback)
            elif both_downtrend and near_ema:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: trend breaks on either timeframe or price moves >1.5% from EMA
            if not (uptrend_4h and uptrend_1d) or abs(close[i] - ema34_1h[i]) / ema34_1h[i] > 0.015:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: trend breaks on either timeframe or price moves >1.5% from EMA
            if not (downtrend_4h and downtrend_1d) or abs(close[i] - ema34_1h[i]) / ema34_1h[i] > 0.015:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_HTF_4h1d_DualTrend_Filter"
timeframe = "1h"
leverage = 1.0