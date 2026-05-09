#!/usr/bin/env python3
# 4h_TRIX_Trend_Filter
# Strategy: TRIX(12) momentum with 1d EMA(50) trend filter
# Long when TRIX crosses above 0 and price > 1d EMA50
# Short when TRIX crosses below 0 and price < 1d EMA50
# Exit when TRIX crosses back through zero
# Uses momentum confirmation with trend filter to avoid counter-trend trades
# Designed for 4h timeframe with selective entries to minimize trade frequency

name = "4h_TRIX_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate TRIX(12): Triple EMA of percentage change
    # TRIX = EMA(EMA(EMA(roc, 12), 12), 12) where roc = (close - close_prev) / close_prev * 100
    roc = np.diff(close, prepend=close[0]) / np.where(close == 0, 1, close) * 100
    
    # Triple EMA smoothing
    ema1 = pd.Series(roc).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = ema3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 36  # Need enough data for triple EMA (12*3)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: TRIX crosses above 0 and price above 1d EMA50 (uptrend filter)
            if trix[i] > 0 and trix[i-1] <= 0 and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below 0 and price below 1d EMA50 (downtrend filter)
            elif trix[i] < 0 and trix[i-1] >= 0 and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses back below 0
            if trix[i] < 0 and trix[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses back above 0
            if trix[i] > 0 and trix[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals