#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_Regime
Hypothesis: 4h TRIX momentum with volume spike and chop regime filter captures
strong trending moves while avoiding whipsaws in range markets. TRIX > 0 with
volume confirmation indicates bullish momentum, TRIX < 0 indicates bearish.
Chop filter (CHOP > 61.8) avoids false signals in ranging markets. Designed
to work in both bull and bear markets by following momentum with regime awareness.
Targets 20-50 trades/year to minimize fee drag on 4h timeframe.
"""
name = "4h_TRIX_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1D data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate TRIX (12,12,12) - triple EMA of ROC
    # ROC = (close - close[12]) / close[12] * 100
    close_series = pd.Series(close)
    roc = close_series.pct_change(12) * 100  # 12-period ROC
    ema1 = roc.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ema3.values
    
    # Calculate Chop Index for regime detection (using 1D data)
    # CHOP = 100 * log10(sum(ATR,14) / (max(high,14) - min(low,14))) / log10(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    chop = 100 * np.log10(pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values / range_14) / np.log10(14)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume filter: current 4h volume > 1.8 x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.8)
    
    # Align TRIX to LTF (no additional delay needed for TRIX)
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX positive, volume spike, and not in chop regime (trending market)
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and 
                volume_filter[i] and chop_aligned[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # Short: TRIX negative, volume spike, and not in chop regime (trending market)
            elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and 
                  volume_filter[i] and chop_aligned[i] < 61.8):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: TRIX crosses zero (momentum reversal)
            if position == 1 and trix_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            elif position == -1 and trix_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals