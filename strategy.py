#!/usr/bin/env python3
# 4h_1d_Trix_ZeroCross_Volume
# Hypothesis: On 4h timeframe, trade TRIX zero-cross signals from daily timeframe with volume confirmation.
# TRIX (Triple Exponential Average) is effective at identifying momentum changes and trend reversals.
# Using daily TRIX reduces noise and provides more reliable signals in both bull and bear markets.
# Volume confirmation filters out weak breakouts. Targets 20-40 trades per year.

name = "4h_1d_Trix_ZeroCross_Volume"
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
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily TRIX (15-period)
    close_1d = df_1d['close'].values
    
    # Triple EMA: EMA(EMA(EMA(close, 15), 15), 15)
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    
    # TRIX = (EMA3 - previous EMA3) / previous EMA3 * 100
    trix_raw = np.zeros_like(ema3)
    trix_raw[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    
    # Align daily TRIX to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix_raw)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long when TRIX crosses above zero with volume confirmation
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and 
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short when TRIX crosses below zero with volume
            elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and 
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TRIX crosses below zero
            if trix_aligned[i] < 0 and trix_aligned[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX crosses above zero
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals