#!/usr/bin/env python3
"""
12H_TRIX_Trend_Volume_Signal
Hypothesis: TRIX (12,20) on 1d timeframe captures momentum with reduced whipsaw; combined with volume spike and 12h timeframe entry.
Works in bull markets via upward TRIX cross and in bear markets via downward TRIX cross.
Volume filter ensures participation only during active market phases, reducing false signals.
Targets 12-37 trades/year to minimize fee drag on 12h timeframe.
"""
name = "12H_TRIX_Trend_Volume_Signal"
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
    
    # Get 1D data for TRIX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate TRIX: triple EMA of log returns
    # EMA1 of close
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA2 of EMA1
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA3 of EMA2
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # Percent change of EMA3
    trix_raw = np.diff(ema3) / ema3[:-1] * 100
    # Smooth with 20-period EMA (signal line)
    trix = pd.Series(trix_raw).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Pad to match original length
    trix = np.concatenate([np.full(33, np.nan), trix])  # 12+12+12+20-3 = 53? Actually: 12+12+12=36, then 20 EMA -> 36+19=55? Simpler: just pad to match
    # Recalculate padding: TRIX raw starts at index 33 (0-based, after 33 periods of 12*3)
    # Then 20-period EMA needs 19 more -> index 33+19=52 is first valid
    trix = np.concatenate([np.full(52, np.nan), trix_raw[20-1:]]) if len(trix_raw) >= 20 else np.full_like(close_1d, np.nan)
    if len(trix) < len(close_1d):
        # Extend if needed
        trix = np.concatenate([trix, np.full(len(close_1d) - len(trix), np.nan)])
    elif len(trix) > len(close_1d):
        trix = trix[:len(close_1d)]
    
    # Volume filter: current volume > 1.8 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    # Align TRIX to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 55  # Ensure TRIX and volume data are ready
    
    for i in range(start_idx, n):
        if np.isnan(trix_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero with volume confirmation
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume confirmation
            elif trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX crosses below zero
            if trix_aligned[i] < 0 and trix_aligned[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX crosses above zero
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals