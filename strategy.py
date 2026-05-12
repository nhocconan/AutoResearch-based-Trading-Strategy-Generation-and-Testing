#!/usr/bin/env python3
# 4h_1d_TRIX_VolumeSpike_Regime
# Hypothesis: TRIX (12-period) on 4h timeframe with volume spike and choppiness regime filter.
# Enters long when TRIX crosses above zero line with volume spike in trending regime (CHOP < 38.2).
# Enters short when TRIX crosses below zero line with volume spike in trending regime (CHOP < 38.2).
# Uses 1d timeframe for TRIX calculation to reduce noise and focus on medium-term momentum.
# Designed for low trade frequency (~50-150 total trades over 4 years) to minimize fee drag.
# Works in bull/bear markets by using TRIX zero-line cross as momentum signal and choppiness filter to avoid ranging markets.

name = "4h_1d_TRIX_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >1.5x 20-period average (on 4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Daily data for TRIX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate TRIX (12-period) on daily close
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - 1 period ago, then percentage change
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix_raw = pd.Series(ema3).pct_change(1).values * 100  # Percentage change
    
    # Choppiness Index (14-period) on daily data for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d_arr[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d_arr[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Absolute close change over 14 periods
    abs_close_chg = np.abs(close_1d_arr - np.roll(close_1d_arr, 14))
    abs_close_chg[0:14] = np.nan  # First 14 values are NaN
    
    # Choppiness Index
    chop = 100 * np.log10(tr_sum / abs_close_chg) / np.log10(14)
    
    # Align all indicators to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix_raw)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(trix_aligned[i]) or
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: TRIX crosses above zero + volume spike + trending regime (CHOP < 38.2)
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and 
                volume_spike[i] and chop_aligned[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + volume spike + trending regime (CHOP < 38.2)
            elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and 
                  volume_spike[i] and chop_aligned[i] < 38.2):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero OR chop regime shifts to ranging (CHOP > 61.8)
            if (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0) or \
               (chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero OR chop regime shifts to ranging (CHOP > 61.8)
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0) or \
               (chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals