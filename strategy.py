#!/usr/bin/env python3
"""
4h_TRIX_Zero_Cross_Volume_Filter
Hypothesis: Use TRIX (triple-smoothed EMA) zero-cross for trend detection with volume confirmation.
TRIX filters out insignificant price movements and is effective in both trending and ranging markets.
Long when TRIX crosses above zero with volume > 1.5x average, short when crosses below zero with volume > 1.5x average.
Exit on opposite TRIX cross. Targets 25-40 trades/year via TRIX smoothing + volume filter.
Works in bull/bear by following momentum with volume confirmation to avoid false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate EMA with proper handling of NaN values."""
    if len(values) < period:
        return np.full_like(values, np.nan)
    result = np.full_like(values, np.nan)
    multiplier = 2 / (period + 1)
    result[period-1] = np.mean(values[:period])
    for i in range(period, len(values)):
        if np.isnan(values[i]):
            result[i] = result[i-1]
        else:
            result[i] = values[i] * multiplier + result[i-1] * (1 - multiplier)
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for TRIX calculation
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate TRIX: triple EMA of close, then ROC
    # EMA1
    ema1 = ema(close_4h, 12)
    # EMA2 of EMA1
    ema2 = ema(ema1, 12)
    # EMA3 of EMA2
    ema3 = ema(ema2, 12)
    
    # TRIX = (EMA3 - previous EMA3) / previous EMA3 * 100
    trix = np.full_like(close_4h, np.nan)
    for i in range(1, len(ema3)):
        if not np.isnan(ema3[i]) and not np.isnan(ema3[i-1]) and ema3[i-1] != 0:
            trix[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
    
    # Align TRIX to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_4h, trix)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, vol_period)  # Need enough data for TRIX and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(trix_aligned[i]) or np.isnan(trix_aligned[i-1]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: TRIX crosses above zero + volume confirmation
            if trix_aligned[i-1] <= 0 and trix_aligned[i] > 0 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero + volume confirmation
            elif trix_aligned[i-1] >= 0 and trix_aligned[i] < 0 and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TRIX crosses below zero
            if trix_aligned[i-1] >= 0 and trix_aligned[i] < 0:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX crosses above zero
            if trix_aligned[i-1] <= 0 and trix_aligned[i] > 0:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TRIX_Zero_Cross_Volume_Filter"
timeframe = "4h"
leverage = 1.0