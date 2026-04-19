#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX(9) momentum with daily volume spike and choppiness regime filter.
# Long when TRIX crosses above zero AND volume spike AND chop > 61.8 (ranging market for mean reversion).
# Short when TRIX crosses below zero AND volume spike AND chop > 61.8.
# Uses TRIX for momentum, volume spike for conviction, chop filter to avoid whipsaw in strong trends.
# Target: 20-50 trades/year per symbol (~80-200 total over 4 years).
name = "4h_TRIX9_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for chop calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate TRIX(9) on close
    # TRIX = EMA(EMA(EMA(close, 9), 9), 9) then % change
    ema1 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix_raw = pd.Series(ema3).pct_change() * 100  # percentage
    trix = trix_raw.fillna(0).values
    
    # Calculate Chop(14) on daily: 100 * log10(sum(ATR(1)) / (max(high) - min(low))) / log10(14)
    # We need at least 14 days of data
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]  # first bar
    atr_1d = pd.Series(tr_1d).rolling(window=1, min_periods=1).sum().values  # ATR(1) = TR
    sum_tr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(sum_tr_14 / (max_high_14 - min_low_14)) / np.log10(14)
    chop_1d = np.where((max_high_14 - min_low_14) == 0, 100, chop_1d)  # avoid div by zero
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(27, 20, 14)  # TRIX needs ~27, volume MA 20, chop 14
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        chop = chop_aligned[i]
        trix_now = trix[i]
        trix_prev = trix[i-1]
        
        # Volume confirmation threshold
        volume_spike = vol > 2.0 * vol_ma
        # Chop filter: chop > 61.8 indicates ranging market (good for mean reversion)
        chop_filter = chop > 61.8
        
        if position == 0:
            # Enter long: TRIX crosses above zero AND volume spike AND chop filter
            if trix_prev <= 0 and trix_now > 0 and volume_spike and chop_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below zero AND volume spike AND chop filter
            elif trix_prev >= 0 and trix_now < 0 and volume_spike and chop_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when TRIX crosses below zero
            if trix_prev >= 0 and trix_now < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when TRIX crosses above zero
            if trix_prev <= 0 and trix_now > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals