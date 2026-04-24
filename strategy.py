#!/usr/bin/env python3
"""
Hypothesis: 12h TRIX + volume spike + choppiness regime filter.
- Long when TRIX crosses above zero AND chop > 61.8 (ranging market) AND volume > 2.0x 20-period average
- Short when TRIX crosses below zero AND chop > 61.8 (ranging market) AND volume > 2.0x 20-period average
- Exit when TRIX crosses zero in opposite direction
- Uses 12h primary timeframe with 1d HTF for choppiness regime filter to target 50-150 trades over 4 years
- TRIX is effective at catching momentum reversals in ranging markets
- Volume confirmation reduces false signals
- Choppiness filter ensures we only trade in ranging conditions where mean reversion works
- Designed to work in both bull and bear markets by focusing on ranging regimes
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
    
    # Calculate TRIX (15,9,9) - triple exponential moving average
    # TRIX = EMA(EMA(EMA(close, 15), 9), 9)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # TRIX = (ema3 - previous ema3) / previous ema3 * 100
    trix_raw = np.zeros_like(ema3)
    trix_raw[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    trix_raw[0] = 0.0
    
    # Signal line: 9-period EMA of TRIX
    trix_signal = pd.Series(trix_raw).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Get 1d data ONCE before loop for choppiness regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Choppiness Index (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], 
                           np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index = 100 * log10(tr_sum / (hh_1d - ll_1d)) / log10(14)
    chop_1d = np.zeros_like(tr_sum)
    denominator = hh_1d - ll_1d
    mask = (denominator > 0) & ~np.isnan(denominator)
    chop_1d[mask] = 100 * np.log10(tr_sum[mask] / denominator[mask]) / np.log10(14)
    
    # Align 1d Choppiness to 12h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation: > 2.0x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(15+9+9, 9, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_signal[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above signal line, chop > 61.8 (ranging), volume confirmation
            if (trix_raw[i] > trix_signal[i] and trix_raw[i-1] <= trix_signal[i-1] and
                chop_1d_aligned[i] > 61.8 and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below signal line, chop > 61.8 (ranging), volume confirmation
            elif (trix_raw[i] < trix_signal[i] and trix_raw[i-1] >= trix_signal[i-1] and
                  chop_1d_aligned[i] > 61.8 and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below signal line
            if trix_raw[i] < trix_signal[i] and trix_raw[i-1] >= trix_signal[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above signal line
            if trix_raw[i] > trix_signal[i] and trix_raw[i-1] <= trix_signal[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_TRIX_ChopRegime_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0