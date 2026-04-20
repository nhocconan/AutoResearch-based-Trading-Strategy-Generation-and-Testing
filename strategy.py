#!/usr/bin/env python3
# 4h_1w_TRIX_Momentum_VolumeFilter
# Hypothesis: Weekly TRIX momentum (3-period ROC of triple-smoothed EMA) with volume filter 
# captures strong momentum moves while avoiding whipsaws. TRIX is effective in both bull and bear 
# markets as it filters out insignificant cycles. Volume ensures institutional participation.
# Target: 15-25 trades/year to minimize fee drag.

name = "4h_1w_TRIX_Momentum_VolumeFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly TRIX: 3-period ROC of triple-smoothed EMA (15-period)
    close_1w = df_1w['close'].values
    
    # Triple EMA smoothing
    ema1 = pd.Series(close_1w).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    
    # TRIX = 3-period ROC of triple-smoothed EMA
    trix = np.zeros_like(ema3)
    trix[15:] = (ema3[15:] - ema3[:-15]) / ema3[:-15] * 100
    
    # Align weekly TRIX to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1w, trix)
    
    # Volume filter: volume > 2.0x 20-period EMA (stringent to reduce trades)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > (vol_ema20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure TRIX calculation is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(trix_aligned[i]) or np.isnan(volume_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero line + volume confirmation
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero line + volume confirmation
            elif trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if TRIX crosses below zero line
            if trix_aligned[i] < 0 and trix_aligned[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if TRIX crosses above zero line
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals