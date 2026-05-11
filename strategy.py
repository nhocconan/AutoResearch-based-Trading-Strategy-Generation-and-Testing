#!/usr/bin/env python3
"""
6h_MarketProfile_ValueArea_Breakout_1wTrend_Filter
Hypothesis: Uses weekly value area (70% POC-based range) as dynamic support/resistance.
Long when price breaks above value area high with weekly uptrend; short when breaks below value area low with weekly downtrend.
Value area adapts to volatility, providing structure in both trending and ranging markets.
Weekly trend filter ensures alignment with higher timeframe momentum.
Target: 20-50 trades/year to minimize fee drag on 6s timeframe.
"""

name = "6h_MarketProfile_ValueArea_Breakout_1wTrend_Filter"
timeframe = "6h"
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
    
    # === WEEKLY Data for Value Area and Trend ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Value Area (70% of volume around POC)
    # Simplified: use weekly range and assume normal distribution
    weekly_range = high_1w - low_1w
    poc_1w = (high_1w + low_1w) / 2  # approximate POC as midpoint
    va_width = 0.7 * weekly_range  # 70% of range
    va_high_1w = poc_1w + va_width / 2
    va_low_1w = poc_1w - va_width / 2
    
    # Weekly EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align to 6h timeframe
    va_high_aligned = align_htf_to_ltf(prices, df_1w, va_high_1w)
    va_low_aligned = align_htf_to_ltf(prices, df_1w, va_low_1w)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(va_high_aligned[i]) or np.isnan(va_low_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above value area high with weekly uptrend
            if (close[i] > va_high_aligned[i] and 
                close[i] > ema20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below value area low with weekly downtrend
            elif (close[i] < va_low_aligned[i] and 
                  close[i] < ema20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below value area low (mean reversion)
            if close[i] < va_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: break above value area high (mean reversion)
            if close[i] > va_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals