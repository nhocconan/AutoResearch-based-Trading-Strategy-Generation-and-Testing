#!/usr/bin/env python3
# 12h_TRIX_ZeroCross_Volume_Spike
# Hypothesis: TRIX (15-period) zero-cross with volume spike on 12h timeframe. TRIX filters noise and captures momentum; volume spike confirms institutional participation. Works in bull/bear via zero-cross direction and avoids whipsaw in chop via TRIX smoothing. Target: 50-150 total trades over 4 years.

name = "12h_TRIX_ZeroCross_Volume_Spike"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # TRIX (15-period) on 12h data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Triple EMA smoothing for TRIX
    close_12h = df_12h['close'].values
    ema1 = pd.Series(close_12h).ewm(span=15, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False).mean().values
    
    # TRIX = 100 * (ema3_today - ema3_yesterday) / ema3_yesterday
    trix_raw = np.zeros_like(ema3)
    trix_raw[1:] = 100 * (ema3[1:] - ema3[:-1]) / ema3[:-1]
    trix_raw[0] = 0.0
    
    trix_12h_aligned = align_htf_to_ltf(prices, df_12h, trix_raw)
    
    # Volume confirmation: volume > 2.0 * 30-period average (strict for fewer trades)
    volume_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for TRIX and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(trix_12h_aligned[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero with volume spike
            if trix_12h_aligned[i] > 0 and trix_12h_aligned[i-1] <= 0 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume spike
            elif trix_12h_aligned[i] < 0 and trix_12h_aligned[i-1] >= 0 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long: exit when TRIX crosses below zero
            if trix_12h_aligned[i] < 0 and trix_12h_aligned[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short: exit when TRIX crosses above zero
            if trix_12h_aligned[i] > 0 and trix_12h_aligned[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals