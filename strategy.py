#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_ChopRegime
Hypothesis: TRIX momentum with volume spike confirmation and choppiness regime filter works in both bull and bear markets by capturing strong momentum moves while avoiding choppy, sideways action. Uses 1d EMA50 as higher timeframe trend filter. Discrete position sizing (0.0, ±0.30) minimizes fee churn. Target: 75-200 trades over 4 years.
"""

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
    
    # TRIX(12): triple EMA of ROC, then smoothed
    # ROC = (close / close.shift(1) - 1) * 100
    roc = np.zeros_like(close)
    roc[1:] = (close[1:] / close[:-1] - 1) * 100
    
    # Triple EMA of ROC
    ema1 = pd.Series(roc).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = ema3  # Already smoothed
    
    # Volume confirmation: volume > 1.8 * 20-period EMA volume (stricter to reduce trades)
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.8 * avg_volume)
    
    # Choppiness Index regime filter: CHOP > 61.8 = range (avoid), CHOP < 38.2 = trending (favor)
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high) - min(low))) / log10(14)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    chop_mask = chop < 40.0  # Only trade when chop < 40 (strong trending)
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(12, 20, 14, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(chop[i]) or np.isnan(ema_50_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        # Discrete position sizing
        base_size = 0.30
        
        # Long logic: TRIX crosses above zero + volume spike + chop regime (trending) + price > 1d EMA50 (uptrend)
        if trix[i] > 0 and trix[i-1] <= 0 and volume_spike[i] and chop_mask[i] and close[i] > ema_50_1d_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: TRIX crosses below zero + volume spike + chop regime (trending) + price < 1d EMA50 (downtrend)
        elif trix[i] < 0 and trix[i-1] >= 0 and volume_spike[i] and chop_mask[i] and close[i] < ema_50_1d_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: TRIX crosses zero in opposite direction
        elif position == 1 and trix[i] < 0 and trix[i-1] >= 0:
            signals[i] = 0.0
            position = 0
        elif position == -1 and trix[i] > 0 and trix[i-1] <= 0:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0