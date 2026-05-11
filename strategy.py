#!/usr/bin/env python3
name = "12h_TRIX_Volume_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for TRIX and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate TRIX (15-period)
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix_raw = np.diff(ema3, prepend=ema3[0]) / ema3 * 100
    trix = pd.Series(trix_raw).ewm(span=9, adjust=False, min_periods=9).mean().values  # Signal line
    
    # Volume confirmation: 20-day average
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_up_1w = close_1w > ema20_1w
    
    # Align indicators to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 30)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(trix_aligned[i]) or 
            np.isnan(vol_ma20_1d_aligned[i]) or
            np.isnan(trend_up_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero + volume confirmation + weekly uptrend
            if (trix_aligned[i] > 0 and 
                trix_aligned[i-1] <= 0 and  # Cross above zero
                volume[i] > 1.3 * vol_ma20_1d_aligned[i] and
                trend_up_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero + volume confirmation + weekly downtrend
            elif (trix_aligned[i] < 0 and 
                  trix_aligned[i-1] >= 0 and  # Cross below zero
                  volume[i] > 1.3 * vol_ma20_1d_aligned[i] and
                  not trend_up_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero or trend changes
            if (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0) or \
               not trend_up_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero or trend changes
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0) or \
               trend_up_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals