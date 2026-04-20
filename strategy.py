#!/usr/bin/env python3
# 12h_1d_TRIX_VolumeSpike_Cross
# Hypothesis: On 12h timeframe, use 1d TRIX (Triple Exponential Average) with zero-line crossovers and volume spike confirmation to capture momentum shifts.
# TRIX filters noise and highlights sustained momentum. Volume spike confirms conviction. Works in bull/bear by following momentum direction.
# Entry: Long when TRIX crosses above zero with volume > 1.5x 20-period average; Short when TRIX crosses below zero with volume spike.
# Exit: Reverse signal or volume drops below average. Targets 15-25 trades/year.

name = "12h_1d_TRIX_VolumeSpike_Cross"
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 18:
        return np.zeros(n)
    
    # Calculate 1d TRIX (15-period EMA of EMA of EMA, then ROC)
    close_1d = df_1d['close'].values
    # First EMA
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Second EMA
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Third EMA
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # TRIX: 1-period ROC of triple EMA
    trix = np.zeros_like(ema3)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    
    # Align TRIX to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(trix_aligned[i-1]) or 
            np.isnan(volume_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero with volume spike
            if (trix_aligned[i-1] <= 0 and trix_aligned[i] > 0 and 
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume spike
            elif (trix_aligned[i-1] >= 0 and trix_aligned[i] < 0 and 
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TRIX crosses below zero or volume drops
            if trix_aligned[i] < 0 or volume[i] < volume_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX crosses above zero or volume drops
            if trix_aligned[i] > 0 or volume[i] < volume_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals