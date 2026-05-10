#!/usr/bin/env python3
"""
12H_1W_1D_CCI_MeanReversion
Hypothesis: Use weekly CCI extremes for mean-reversion in 12h timeframe.
Buy when weekly CCI < -100 and price touches 1d lower Bollinger Band (mean-reversion in oversold).
Sell when weekly CCI > 100 and price touches 1d upper Bollinger Band.
Uses weekly structure to identify extremes and daily Bollinger for entry timing.
Works in both bull and bear markets by fading extremes. Target: 20-40 trades/year per symbol.
"""

name = "12H_1W_1D_CCI_MeanReversion"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d and 1w data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly CCI(20)
    tp_1w = (high_1w + low_1w + close_1w) / 3.0
    sma_tp_1w = pd.Series(tp_1w).rolling(window=20, min_periods=20).mean().values
    mad_1w = pd.Series(tp_1w).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    cci_1w = (tp_1w - sma_tp_1w) / (0.015 * mad_1w)
    cci_1w = np.where(mad_1w == 0, 0, cci_1w)  # avoid division by zero
    
    # Daily Bollinger Bands(20,2)
    sma_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_1d + 2 * std_1d
    lower_bb_1d = sma_1d - 2 * std_1d
    
    # Align to 12h
    cci_aligned = align_htf_to_ltf(prices, df_1w, cci_1w)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(cci_aligned[i]) or np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: weekly CCI < -100 (oversold) and price touches lower BB
            if cci_aligned[i] < -100 and close[i] <= lower_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly CCI > 100 (overbought) and price touches upper BB
            elif cci_aligned[i] > 100 and close[i] >= upper_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly CCI > -50 (recovering from oversold) or price reaches middle BB
            middle_bb_1d = sma_1d
            middle_aligned = align_htf_to_ltf(prices, df_1d, middle_bb_1d)
            if cci_aligned[i] > -50 or close[i] >= middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly CCI < 50 (declining from overbought) or price reaches middle BB
            middle_bb_1d = sma_1d
            middle_aligned = align_htf_to_ltf(prices, df_1d, middle_bb_1d)
            if cci_aligned[i] < 50 or close[i] <= middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals