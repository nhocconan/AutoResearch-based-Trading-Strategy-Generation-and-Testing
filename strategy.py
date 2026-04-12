#!/usr/bin/env python3
"""
1h_4h_1d_trend_continuation_v1
Hypothesis: 1-hour strategy using 4h trend filter and 1d momentum filter.
Enters long when price closes above 4h EMA20 and 1d ROC > 0, short when below 4h EMA20 and 1d ROC < 0.
Uses 1h for entry timing only to reduce whipsaw. Target: 15-30 trades/year (60-120 total) to minimize fee drag.
Works in bull/bear by following higher timeframe trend with momentum confirmation.
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA20 for trend direction
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Get 1d data for momentum filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # 1d ROC(5) for momentum
    close_1d = df_1d['close'].values
    roc5_1d = np.zeros_like(close_1d)
    roc5_1d[5:] = (close_1d[5:] - close_1d[:-5]) / close_1d[:-5] * 100
    roc5_1d_aligned = align_htf_to_ltf(prices, df_1d, roc5_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(roc5_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: 4h trend + 1d momentum
        long_entry = close[i] > ema20_4h_aligned[i] and roc5_1d_aligned[i] > 0
        short_entry = close[i] < ema20_4h_aligned[i] and roc5_1d_aligned[i] < 0
        
        # Exit conditions: trend or momentum reversal
        long_exit = close[i] < ema20_4h_aligned[i] or roc5_1d_aligned[i] <= 0
        short_exit = close[i] > ema20_4h_aligned[i] or roc5_1d_aligned[i] >= 0
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_trend_continuation_v1"
timeframe = "1h"
leverage = 1.0