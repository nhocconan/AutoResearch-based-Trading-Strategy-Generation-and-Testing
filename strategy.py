#!/usr/bin/env python3
"""
4h_TRIX_ZeroCross_VolumeSpike_ChopFilter
Hypothesis: 4h TRIX zero cross with volume spike and chop regime filter. 
Long when TRIX crosses above zero with volume > 2x average and CHOP > 61.8 (range) for mean reversion bounce.
Short when TRIX crosses below zero with volume > 2x average and CHOP > 61.8 for mean reversion fade.
Exit on opposite TRIX cross or CHOP < 38.2 (trend) to avoid whipsaw.
Uses discrete sizing (0.25) to minimize fee drag. Target: 20-50 trades/year.
Works in bull via momentum continuation, in bear via mean reversion in ranging markets.
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
    
    # Get 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) for chop
    atr_1d = np.zeros(len(close_1d))
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                  np.abs(low_1d[1:] - close_1d[:-1])))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate chop: ATR(14) / (HHV(14) - LLV(14)) * 100
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = np.where((hh_14 - ll_14) > 0, (atr_1d / (hh_14 - ll_14)) * 100, 50)
    
    # Align chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate TRIX(12,9,9) on 4h close
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = pd.Series(ema3).pct_change(periods=9) * 100
    trix_values = trix.values
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for TRIX (12+12+12+9 = 45) + chop (30) + vol (20)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_values[i]) or np.isnan(trix_values[i-1]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # TRIX zero cross signals
        trix_cross_up = (trix_values[i-1] <= 0) and (trix_values[i] > 0)
        trix_cross_down = (trix_values[i-1] >= 0) and (trix_values[i] < 0)
        
        if position == 0:
            # Long: TRIX crosses up with volume spike and chop > 61.8 (range)
            long_signal = trix_cross_up and vol_spike[i] and (chop_aligned[i] > 61.8)
            # Short: TRIX crosses down with volume spike and chop > 61.8 (range)
            short_signal = trix_cross_down and vol_spike[i] and (chop_aligned[i] > 61.8)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: TRIX cross down OR chop < 38.2 (trend) to avoid whipsaw
            exit_signal = trix_cross_down or (chop_aligned[i] < 38.2)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: TRIX cross up OR chop < 38.2 (trend)
            exit_signal = trix_cross_up or (chop_aligned[i] < 38.2)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_TRIX_ZeroCross_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0