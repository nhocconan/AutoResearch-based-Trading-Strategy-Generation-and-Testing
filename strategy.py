#!/usr/bin/env python3
"""
12h_TRIX_ZeroCross_TrendFilter
Hypothesis: TRIX (12-period) zero-cross signals on 12h timeframe, filtered by 1d EMA50 trend and volume spikes, capture momentum while avoiding whipsaws in both bull and bear markets. TRIX filters noise better than MACD, and 12h timeframe reduces trade frequency to manageable levels.
"""

name = "12h_TRIX_ZeroCross_TrendFilter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate TRIX (12-period) on 12h close prices
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - then percentage change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean()
    trix = (ema3 / ema3.shift(1) - 1) * 100  # Percentage change
    trix_values = trix.values
    
    # Get 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (volume > 1.5x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need TRIX calculation (needs ~36 for 3 EMAs) + EMA50 + volume MA
    start_idx = max(40, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(trix_values[i]) or 
            np.isnan(trix_values[i-1]) or  # Need previous value for zero-cross
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend_1d = close[i] > ema_50_1d_aligned[i]
        downtrend_1d = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation (>1.5x average volume)
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # TRIX zero-cross signals
        trix_cross_up = trix_values[i-1] <= 0 and trix_values[i] > 0
        trix_cross_down = trix_values[i-1] >= 0 and trix_values[i] < 0
        
        if position == 0:
            # Long entry: TRIX crosses up from below zero + uptrend + volume confirmation
            if trix_cross_up and uptrend_1d and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX crosses down from above zero + downtrend + volume confirmation
            elif trix_cross_down and downtrend_1d and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses down or trend breaks
            if trix_cross_down or not uptrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses up or trend breaks
            if trix_cross_up or not downtrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals