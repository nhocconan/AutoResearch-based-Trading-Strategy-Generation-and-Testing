#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_TrendFilter_v1
Hypothesis: Use TRIX momentum on 1d timeframe combined with volume spike and price position relative to TRIX signal line to capture trend reversals in both bull and bear markets. Entry when TRIX crosses above/below signal line with volume confirmation, exit on opposite cross. Designed for fewer trades (<50/year) with strong edge in ranging and trending conditions.
"""

name = "4h_TRIX_VolumeSpike_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for TRIX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    
    # Calculate TRIX: triple EMA of ROC
    # ROC = (close - close.shift(1)) / close.shift(1)
    roc = np.diff(daily_close, prepend=daily_close[0]) / np.where(daily_close == 0, 1e-10, daily_close)
    roc[0] = 0  # first ROC is zero
    
    # Triple EMA of ROC
    ema1 = pd.Series(roc).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = ema3 * 100  # scale for readability
    
    # Signal line: EMA of TRIX
    signal_line = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Align TRIX and signal line to 4h timeframe (with 1-day delay for completed bar)
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix, additional_delay_bars=1)
    signal_aligned = align_htf_to_ltf(prices, df_1d, signal_line, additional_delay_bars=1)
    
    # Get 4h volume for confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix_aligned[i]) or 
            np.isnan(signal_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: TRIX crosses above signal line with volume spike
            if (trix_aligned[i] > signal_aligned[i] and 
                trix_aligned[i-1] <= signal_aligned[i-1] and 
                vol_ratio[i] > 3.0):  # Volume spike threshold
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX crosses below signal line with volume spike
            elif (trix_aligned[i] < signal_aligned[i] and 
                  trix_aligned[i-1] >= signal_aligned[i-1] and 
                  vol_ratio[i] > 3.0):  # Volume spike threshold
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX crosses below signal line
            if trix_aligned[i] < signal_aligned[i] and trix_aligned[i-1] >= signal_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX crosses above signal line
            if trix_aligned[i] > signal_aligned[i] and trix_aligned[i-1] <= signal_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals