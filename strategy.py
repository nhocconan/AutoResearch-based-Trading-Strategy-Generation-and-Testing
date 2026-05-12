#!/usr/bin/env python3
"""
4h_TRIX_Volume_Spike_Chop_Filter
Hypothesis: On 4h timeframe, TRIX (Triple Exponential Average) crossing above/below zero 
with volume > 2x 20-period average and Choppiness Index > 61.8 (ranging market) 
generates mean-reversion signals. TRIX captures momentum, volume confirms strength, 
and Choppiness filters for ranging conditions where mean reversion works best.
Targets 20-50 trades/year (80-200 total over 4 years) with moderate turnover.
Works in both bull and bear markets by adapting to ranging conditions.
"""

name = "4h_TRIX_Volume_Spike_Chop_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for Choppiness Index (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate TRIX (15-period)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = np.diff(ema3, prepend=ema3[0]) / ema3 * 100

    # Calculate Choppiness Index (14-period) on daily data
    atr_1d = []
    tr_1d = []
    for i in range(len(close_1d)):
        if i == 0:
            tr = high_1d[i] - low_1d[i]
        else:
            tr = max(high_1d[i] - low_1d[i], 
                     abs(high_1d[i] - close_1d[i-1]), 
                     abs(low_1d[i] - close_1d[i-1]))
        tr_1d.append(tr)
    
    # Calculate ATR with smoothing
    atr_1d = np.array(tr_1d)
    for i in range(1, len(atr_1d)):
        atr_1d[i] = (atr_1d[i-1] * 13 + atr_1d[i]) / 14
    
    # Chop calculation
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr_14 / (max_hh - min_ll)) / np.log10(14)
    chop = np.where((max_hh - min_ll) == 0, 50, chop)  # Avoid division by zero
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)

    # Volume confirmation: 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Get aligned values
        chop_val = chop_aligned[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if np.isnan(chop_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Chop filter: only trade when market is ranging (Chop > 61.8)
        if chop_val <= 61.8:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + volume spike
            if trix[i] > 0 and trix[i-1] <= 0 and volume[i] > vol_avg_val * 2.0:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + volume spike
            elif trix[i] < 0 and trix[i-1] >= 0 and volume[i] > vol_avg_val * 2.0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero
            if trix[i] < 0 and trix[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero
            if trix[i] > 0 and trix[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals