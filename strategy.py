#!/usr/bin/env python3
"""
12h_TRIX_VolumeSpike_ChopRegime_v1
Hypothesis: TRIX momentum combined with volume spike and Choppiness Index regime filter captures trending moves while avoiding whipsaws in ranging markets.
TRIX filters noise, volume spike confirms conviction, Choppiness Index (>61.8) triggers mean-reversion logic, (<38.2) triggers trend-following.
Designed for low trade frequency (12-25/year) on 12h timeframe to minimize fee drag and improve generalization across bull/bear regimes.
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
    
    # Get 1d data for TRIX and Choppiness Index (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    # TRIX calculation: triple EMA of ROC
    # ROC = (close - close_prev) / close_prev
    roc = np.diff(close_1d, prepend=close_1d[0]) / close_1d
    # Three-fold exponential smoothing
    ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = ema3  # Already in percentage terms
    
    # Align TRIX to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Choppiness Index calculation
    # ATR(14) = average true range over 14 periods
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Max/min high-low over 14 periods
    max_hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(atr14) / (max_hh - min_ll)) / log10(14)
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    denom = max_hh - min_ll
    chop = 100 * np.log10(atr_sum / denom) / np.log10(14)
    chop[denom == 0] = 50  # Avoid division by zero
    
    # Align Chop to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike detection: volume > 2.0 * 20-period average (on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(trix_aligned[i]) or 
            np.isnan(chop_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        trix_val = trix_aligned[i]
        chop_val = chop_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: TRIX positive AND volume spike AND trending regime (Chop < 38.2)
            if trix_val > 0 and vol_spike and chop_val < 38.2:
                signals[i] = 0.25
                position = 1
            # Short: TRIX negative AND volume spike AND trending regime (Chop < 38.2)
            elif trix_val < 0 and vol_spike and chop_val < 38.2:
                signals[i] = -0.25
                position = -1
            # Long mean-reversion: TRIX negative AND volume spike AND ranging regime (Chop > 61.8)
            elif trix_val < 0 and vol_spike and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Short mean-reversion: TRIX positive AND volume spike AND ranging regime (Chop > 61.8)
            elif trix_val > 0 and vol_spike and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: TRIX crosses zero OR Chop enters extreme ranging (avoid whipsaw)
            if trix_val <= 0 or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: TRIX crosses zero OR Chop enters extreme ranging
            if trix_val >= 0 or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_TRIX_VolumeSpike_ChopRegime_v1"
timeframe = "12h"
leverage = 1.0