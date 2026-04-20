#!/usr/bin/env python3
# 12h_Trix_Zero_Cross_Volume
# Hypothesis: On 12h timeframe, use TRIX (15-period) zero cross with volume confirmation.
# TRIX filters noise and captures momentum; volume confirms strength.
# Works in bull (rides trends) and bear (captures reversals) due to momentum focus.
# Targets 15-30 trades per year.

name = "12h_Trix_Zero_Cross_Volume"
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
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily TRIX: EMA(EMA(EMA(close, 15), 15), 15) - 1-period percent change
    close_series = pd.Series(df_1d['close'].values)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix_raw = ((ema3 / ema3.shift(1)) - 1) * 100  # percent change
    trix = trix_raw.values
    
    # Align daily TRIX to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long when TRIX crosses above zero with volume confirmation
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and volume[i] > 1.5 * volume_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short when TRIX crosses below zero with volume
            elif trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and volume[i] > 1.5 * volume_ma[i]:
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