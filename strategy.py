#!/usr/bin/env python3
# 12h_Trix_VolumeSpike_TrendFilter
# Hypothesis: TRIX (triple exponential moving average) with volume spike and daily trend filter captures momentum shifts in both bull and bear markets.
# Timeframe: 12h, uses 1d trend filter for multi-timeframe alignment.
# Low trade frequency (~15-30/year) via strict TRIX zero-cross + volume + trend confluence.
# Long: TRIX crosses above 0 with volume > 1.8x average and daily uptrend (close > EMA34).
# Short: TRIX crosses below 0 with volume > 1.8x average and daily downtrend (close < EMA34).
# Exit: Opposite TRIX cross or trend failure.

timeframe = "12h"
name = "12h_Trix_VolumeSpike_TrendFilter"
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
    
    # TRIX calculation: triple EMA of log returns
    # TRIX = (EMA3 of log(close) - previous) / previous * 100
    log_close = np.log(close)
    
    # First EMA
    ema1 = pd.Series(log_close).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Second EMA
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Third EMA
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    
    # TRIX: percentage change of triple EMA
    trix = np.zeros_like(ema3)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Daily trend filter: EMA34
    d_close = df_1d['close'].values
    ema_34_1d = pd.Series(d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 1.8x average volume (12-period = 6 days on 12h chart)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(15, 12)  # Ensure we have TRIX and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above 0 with volume spike and daily uptrend
            if trix[i] > 0 and trix[i-1] <= 0 and volume[i] > 1.8 * vol_ma[i] and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below 0 with volume spike and daily downtrend
            elif trix[i] < 0 and trix[i-1] >= 0 and volume[i] > 1.8 * vol_ma[i] and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX crosses below 0 or trend failure
            if trix[i] < 0 and trix[i-1] >= 0 or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses above 0 or trend failure
            if trix[i] > 0 and trix[i-1] <= 0 or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals