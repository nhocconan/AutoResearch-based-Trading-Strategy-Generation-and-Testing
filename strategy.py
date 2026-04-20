#!/usr/bin/env python3
# 4h_TRIX_Volume_Spike_Regime
# Hypothesis: TRIX momentum with volume spike and chop regime filter captures
# trend changes in both bull and bear markets. TRIX zero-cross signals momentum shifts,
# volume confirms institutional participation, and chop filter avoids whipsaws in ranging markets.
# Position size 0.25 to manage risk. Target: 25-40 trades/year (100-160 total) to minimize fee drag.

name = "4h_TRIX_Volume_Spike_Regime"
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
    
    # Get 1D data for TRIX and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate TRIX (15-period triple EMA)
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean()
    trix_raw = 100 * (ema3 / ema3.shift(1) - 1)
    trix = trix_raw.values
    
    # Calculate 1-day ATR (14-period) for chop filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Chop filter: high-low range / ATR
    chop_denom = np.where(atr_1d != 0, atr_1d, np.nan)
    chop = (high_1d - low_1d) / chop_denom * 100
    chop = chop  # Already aligned to 1d
    
    # Volume spike: volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)
    
    # Align 1D indicators to 4H
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure TRIX and volume MA are calculated
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero + volume spike + chop < 61.8 (trending)
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and volume_spike[i] and chop_aligned[i] < 61.8:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero + volume spike + chop < 61.8 (trending)
            elif trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and volume_spike[i] and chop_aligned[i] < 61.8:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if TRIX crosses below zero or chop increases (range)
            if trix_aligned[i] < 0 or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if TRIX crosses above zero or chop increases (range)
            if trix_aligned[i] > 0 or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals