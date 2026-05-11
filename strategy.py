#!/usr/bin/env python3
name = "6h_1d_TRIX_Volume"
timeframe = "6h"
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
    
    # Get daily data for TRIX and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Calculate TRIX (12-period EMA of EMA of EMA of close, then ROC)
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # Rate of change of the triple EMA
    trix_raw = np.zeros(len(ema3))
    for i in range(len(ema3)):
        if i < 12:
            trix_raw[i] = np.nan
        else:
            trix_raw[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix_raw).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Daily volume moving average (20-period)
    vol_ma20 = np.zeros(len(vol_1d))
    for i in range(len(vol_1d)):
        if i < 20:
            vol_ma20[i] = np.mean(vol_1d[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(vol_1d[i-19:i+1])
    
    # Align indicators to 6h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix_raw)
    trix_signal_aligned = align_htf_to_ltf(prices, df_1d, trix_signal)
    vol_ma20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(trix_aligned[i]) or 
            np.isnan(trix_signal_aligned[i]) or
            np.isnan(vol_ma20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above signal line + volume confirmation
            if (trix_aligned[i] > trix_signal_aligned[i] and 
                trix_aligned[i-1] <= trix_signal_aligned[i-1] and
                volume[i] > 1.5 * vol_ma20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below signal line + volume confirmation
            elif (trix_aligned[i] < trix_signal_aligned[i] and 
                  trix_aligned[i-1] >= trix_signal_aligned[i-1] and
                  volume[i] > 1.5 * vol_ma20_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below signal line
            if (trix_aligned[i] < trix_signal_aligned[i] and 
                trix_aligned[i-1] >= trix_signal_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above signal line
            if (trix_aligned[i] > trix_signal_aligned[i] and 
                trix_aligned[i-1] <= trix_signal_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals