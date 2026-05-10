#!/usr/bin/env python3
"""
12h_1d_TP_SL_Strategy
Hypothesis: Price respects 1-day take-profit (TP) and stop-loss (SL) levels derived from prior day's range. 
In bull markets, buy near prior day's low with target at prior day's high. 
In bear markets, sell near prior day's high with target at prior day's low. 
Uses 12h timeframe for entry timing and 1d for structure. 
Target: 20-40 trades/year to minimize fee drag.
"""

name = "12h_1d_TP_SL_Strategy"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 2:
        return np.zeros(n)
    
    # Get 1-day data for TP/SL levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate TP and SL: TP = prior day's high, SL = prior day's low
    tp_1d = high_1d  # Take profit level
    sl_1d = low_1d   # Stop loss level
    
    # Align 1-day TP/SL to 12h timeframe (with 1-bar delay for completed 1d bar)
    tp_1d_aligned = align_htf_to_ltf(prices, df_1d, tp_1d)
    sl_1d_aligned = align_htf_to_ltf(prices, df_1d, sl_1d)
    
    # 12h data for signal generation
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need at least 2 days of data for TP/SL
    start_idx = 1
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(tp_1d_aligned[i]) or 
            np.isnan(sl_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price near prior day's low (SL level) with bullish bias
            # Enter long when price is within 0.5% of SL level and closing above open
            near_sl = abs((close[i] - sl_1d_aligned[i]) / close[i]) < 0.005 if sl_1d_aligned[i] > 0 else False
            bullish_bias = close[i] > prices['open'].values[i]
            
            if near_sl and bullish_bias:
                signals[i] = 0.25
                position = 1
            # Short entry: price near prior day's high (TP level) with bearish bias
            # Enter short when price is within 0.5% of TP level and closing below open
            near_tp = abs((tp_1d_aligned[i] - close[i]) / close[i]) < 0.005 if tp_1d_aligned[i] > 0 else False
            bearish_bias = close[i] < prices['open'].values[i]
            
            if near_tp and bearish_bias:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches TP level or breaks below SL
            if close[i] >= tp_1d_aligned[i] or close[i] <= sl_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches SL level or breaks above TP
            if close[i] <= sl_1d_aligned[i] or close[i] >= tp_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals