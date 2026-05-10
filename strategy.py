#!/usr/bin/env python3
"""
4h_TRIX_Volume_Spike_Regime
Hypothesis: TRIX (12-period) crossover with volume spike and chop regime filter.
TRIX filters noise and identifies momentum shifts. Works in both bull and bear markets by
combining momentum with volume confirmation and regime filtering (chop > 61.8 = range, chop < 38.2 = trend).
Designed for ~25-35 trades/year on 4h timeframe to avoid excessive fee drag.
"""

name = "4h_TRIX_Volume_Spike_Regime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate chop regime (using 1d data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range and ATR(14)
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # Align with index
    
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Chop = 100 * log15(sum(ATR14,14) / (max(high,14) - min(low,14)))
    atr_sum = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(15)
    
    chop_regime = chop > 61.8  # True = ranging (mean revert), False = trending
    
    # Align chop regime to 4h timeframe
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime.astype(float))
    
    # Calculate TRIX (12-period) on 4h data
    close = prices['close'].values
    
    # EMA1
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA2 of EMA1
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA3 of EMA2
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # TRIX = 100 * (EMA3 - previous EMA3) / previous EMA3
    trix = np.concatenate([[np.nan], 100 * (ema3[1:] - ema3[:-1]) / ema3[:-1]])
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Volume filter: current volume > 1.8x 20-period EMA
    volume = prices['volume'].values
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need chop regime (50 days), TRIX components (12+9=21), volume EMA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(trix[i]) or 
            np.isnan(trix_signal[i]) or
            np.isnan(chop_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above signal AND chop regime is trending (chop < 38.2) AND volume spike
            if trix[i] > trix_signal[i] and trix[i-1] <= trix_signal[i-1] and chop_regime_aligned[i] == False and volume_filter[i]:
                signals[i] = 0.30
                position = 1
            # Short: TRIX crosses below signal AND chop regime is trending (chop < 38.2) AND volume spike
            elif trix[i] < trix_signal[i] and trix[i-1] >= trix_signal[i-1] and chop_regime_aligned[i] == False and volume_filter[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below signal OR chop regime becomes ranging (chop > 61.8)
            if trix[i] < trix_signal[i] or chop_regime_aligned[i] == True:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: TRIX crosses above signal OR chop regime becomes ranging (chop > 61.8)
            if trix[i] > trix_signal[i] or chop_regime_aligned[i] == True:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals