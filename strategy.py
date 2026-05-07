#!/usr/bin/env python3
# 4h_TRIX_VolumeSpike_1dTrend_Regime
# Hypothesis: TRIX momentum with volume spike confirmation and 1d trend filter on 4h timeframe.
# Enters long when TRIX crosses above zero, volume > 2x average, and price above 1d EMA50 (uptrend).
# Enters short when TRIX crosses below zero, volume > 2x average, and price below 1d EMA50 (downtrend).
# Uses 4h TRIX crossovers for entries and opposite TRIX cross for exits to limit holding periods.
# Designed to work in both bull and bear markets via 1d trend filter, with volume confirmation to avoid false signals.
# Target: 20-40 trades/year to stay under fee drag limits.

name = "4h_TRIX_VolumeSpike_1dTrend_Regime"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate TRIX (15-period EMA of EMA of EMA of ROC)
    # ROC(1) = (close[t] - close[t-1]) / close[t-1]
    roc = np.diff(close) / close[:-1]
    roc = np.concatenate([[np.nan], roc])  # align length
    
    # Three-fold EMA smoothing
    ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    
    # Align TRIX to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike detection: 2.0x average volume (20-period for stability)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 45)  # TRIX needs ~45 periods to stabilize (15*3)
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(trix_aligned[i-1]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero, price above 1d EMA50 (uptrend), volume spike
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero, price below 1d EMA50 (downtrend), volume spike
            elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX crosses below zero (opposite signal)
            if trix_aligned[i] < 0 and trix_aligned[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses above zero (opposite signal)
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals