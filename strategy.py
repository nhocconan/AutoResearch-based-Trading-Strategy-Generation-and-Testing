#!/usr/bin/env python3
"""
4h_TRIX_ZeroCross_VolumeSpike_ChopFilter
Hypothesis: 4h TRIX (12,20) zero-cross with volume spike and choppiness regime filter.
Goes long when TRIX crosses above zero with volume > 2.0x 20-period average and CHOP > 61.8 (range),
short when TRIX crosses below zero with volume spike and CHOP > 61.8.
Exit on opposite TRIX cross or CHOP < 38.2 (trend regime).
Uses discrete sizing (0.25) to minimize fees. Target: 20-35 trades/year.
Works in bull via momentum continuation, in bear via mean reversion in range markets.
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
    
    # Calculate TRIX (12,20) - Triple EMA of ROC
    # ROC(12) = (close / close.shift(12) - 1) * 100
    roc = np.zeros_like(close)
    for i in range(12, n):
        roc[i] = (close[i] / close[i-12] - 1) * 100
    
    # EMA1 of ROC
    ema1 = pd.Series(roc).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA2 of EMA1
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA3 of EMA2 = TRIX
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = ema3  # Already in percentage form
    
    # Calculate Choppiness Index (CHOP) - using 14-period
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Sum of TR over 14 periods
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # CHOP = 100 * log10(atr14 / (hh14 - ll14)) / log10(14)
    chop = np.zeros(n)
    for i in range(n):
        if hh14[i] != ll14[i] and not np.isnan(atr14[i]) and atr14[i] > 0:
            chop[i] = 100 * np.log10(atr14[i] / (hh14[i] - ll14[i])) / np.log10(14)
        else:
            chop[i] = 50  # Neutral when range is zero
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix[i]) or np.isnan(chop[i]) or np.isnan(vol_ma_20[i]) or
            i == 0):  # Need previous values for crossover
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # TRIX zero-cross signals
        trix_cross_up = (trix[i-1] <= 0 and trix[i] > 0)
        trix_cross_down = (trix[i-1] >= 0 and trix[i] < 0)
        
        if position == 0:
            # Long: TRIX crosses above zero, volume spike, chop > 61.8 (range)
            long_signal = trix_cross_up and vol_spike[i] and (chop[i] > 61.8)
            # Short: TRIX crosses below zero, volume spike, chop > 61.8 (range)
            short_signal = trix_cross_down and vol_spike[i] and (chop[i] > 61.8)
            
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
            # Exit when TRIX crosses below zero OR chop < 38.2 (trend regime)
            exit_signal = trix_cross_down or (chop[i] < 38.2)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when TRIX crosses above zero OR chop < 38.2 (trend regime)
            exit_signal = trix_cross_up or (chop[i] < 38.2)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_TRIX_ZeroCross_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0