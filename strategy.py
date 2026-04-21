#!/usr/bin/env python3
"""
12h_1w_MovingAverage_Crossover_Trend_Filter
Hypothesis: Price crossing above/below the 50-period EMA on 12h timeframe, 
filtered by weekly EMA200 trend, with volume confirmation, captures strong 
trends while avoiding whipsaws. Weekly trend filter ensures we only trade 
in the direction of higher timeframe momentum, working in both bull and 
bear markets. Target 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = np.zeros_like(close_1w)
    ema200_1w[0] = close_1w[0]
    alpha = 2.0 / (200 + 1)
    for i in range(1, len(close_1w)):
        ema200_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema200_1w[i-1]
    
    # Align weekly EMA200 to 12h timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # 12h data for EMA50 calculation
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50
    ema50 = np.zeros(n)
    ema50[0] = close[0]
    alpha_50 = 2.0 / (50 + 1)
    for i in range(1, n):
        ema50[i] = alpha_50 * close[i] + (1 - alpha_50) * ema50[i-1]
    
    # Volume filter: volume > 1.3x 20-period average
    volume_avg = np.zeros(n)
    for i in range(n):
        if i < 20:
            volume_avg[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            volume_avg[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.3 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(ema200_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema50_val = ema50[i]
        ema200 = ema200_1w_aligned[i]
        vol_confirm = volume_filter[i]
        
        if position == 0:
            # Long: price crosses above EMA50 with volume confirmation in uptrend (price > weekly EMA200)
            if price > ema50_val and vol_confirm and price > ema200:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price crosses below EMA50 with volume confirmation in downtrend (price < weekly EMA200)
            elif price < ema50_val and vol_confirm and price < ema200:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below EMA50 or trend breaks
            if price < ema50_val or price < ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA50 or trend breaks
            if price > ema50_val or price > ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1w_MovingAverage_Crossover_Trend_Filter"
timeframe = "12h"
leverage = 1.0