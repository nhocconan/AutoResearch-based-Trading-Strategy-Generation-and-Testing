#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_ChopFilter
Hypothesis: On 4h timeframe, enter long when TRIX crosses above zero with volume spike in low-chop regime (trending market).
Enter short when TRIX crosses below zero with volume spike in low-chop regime.
Exit when TRIX crosses back through zero or chop increases.
Uses 1d HTF for chop regime filter to avoid false signals in ranging markets.
Designed for low trade frequency (~25-40/year) with strong edge in both bull and bear markets via momentum + regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate TRIX on primary 4h timeframe
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) then % change
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = 100 * (pd.Series(ema3).pct_change(1).values)
    
    # Get 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Chopiness Index on 1d (14-period)
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_1d = np.zeros(len(close_1d))
    c_prev_1d = np.concatenate([[close_1d[0]], close_1d[:-1]])
    for i in range(len(close_1d)):
        tr = true_range(high_1d[i], low_1d[i], c_prev_1d[i])
        if i == 0:
            atr_1d[i] = tr
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr) / 14  # Wilder's smoothing
    
    # Sum of true ranges over 14 periods
    sum_tr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Chopiness Index: 100 * log10(sum_tr_14 / (ATR * 14)) / log10(14)
    chop = 100 * np.log10(sum_tr_14 / (atr_1d * 14)) / np.log10(14)
    
    # Align TRIX and Chop to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=0)  # Chop uses completed 1d bar
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: only trade when chop < 38.2 (strong trending regime)
        in_trend_regime = chop_aligned[i] < 38.2
        
        if position == 0:
            if in_trend_regime and vol_spike[i]:
                # Long: TRIX crosses above zero
                if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0:
                    signals[i] = 0.25
                    position = 1
                # Short: TRIX crosses below zero
                elif trix_aligned[i] < 0 and trix_aligned[i-1] >= 0:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long position
            signals[i] = 0.25
            # Exit: TRIX crosses below zero OR chop increases (trend weakening)
            if trix_aligned[i] < 0 or chop_aligned[i] >= 38.2:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short position
            signals[i] = -0.25
            # Exit: TRIX crosses above zero OR chop increases (trend weakening)
            if trix_aligned[i] > 0 or chop_aligned[i] >= 38.2:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0