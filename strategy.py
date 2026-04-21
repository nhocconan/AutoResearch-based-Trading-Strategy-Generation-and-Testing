#!/usr/bin/env python3
"""
1d_RangeBound_MeanReversion_Bollinger
Hypothesis: Trade mean-reversion within range-bound markets using Bollinger Bands with Bollinger Band Width percentile as regime filter.
In range-bound markets (BBW < 30th percentile), buy at lower band and sell at upper band. 
In trending markets (BBW >= 30th percentile), no trades to avoid whipsaw.
Designed for 1d timeframe to target 20-50 trades/year with high-conviction entries.
Works in bull markets by capturing pullbacks in uptrends and in bear markets by capturing bounces in downtrends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for Bollinger Bands and BBW
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Bollinger Bands (20, 2)
    sma20 = np.zeros_like(close_1d)
    std20 = np.zeros_like(close_1d)
    for i in range(20, len(close_1d)):
        sma20[i] = np.mean(close_1d[i-20:i])
        std20[i] = np.std(close_1d[i-20:i])
    
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    
    # Bollinger Band Width
    bbw = (upper - lower) / sma20
    
    # Align to 1d timeframe (no additional delay needed for BB/BBW)
    sma20_aligned = align_htf_to_ltf(prices, df_1d, sma20)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    bbw_aligned = align_htf_to_ltf(prices, df_1d, bbw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(sma20_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or np.isnan(bbw_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: range-bound if BBW < 30th percentile of lookback
        if i >= 50:
            bbw_percentile = np.percentile(bbw_aligned[:i+1], 30)
            range_bound = bbw_aligned[i] < bbw_percentile
        else:
            range_bound = True  # Allow trades during warmup
        
        price = prices['close'].iloc[i]
        
        if position == 0 and range_bound:
            # Long at lower band
            if price <= lower_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short at upper band
            elif price >= upper_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches middle band or regime changes to trending
            if price >= sma20_aligned[i] or not range_bound:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches middle band or regime changes to trending
            if price <= sma20_aligned[i] or not range_bound:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_RangeBound_MeanReversion_Bollinger"
timeframe = "1d"
leverage = 1.0