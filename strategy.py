#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_Regime
Strategy: Use TRIX momentum with volume spike confirmation and choppiness regime filter.
Long: TRIX > 0 + volume > 1.5x average + CHOP > 61.8 (ranging market)
Short: TRIX < 0 + volume > 1.5x average + CHOP > 61.8 (ranging market)
Exit: TRIX crosses zero or volatility breaks out (CHOP < 38.2)
Position size: 0.25
Designed to capture mean-reversion moves in ranging markets with momentum confirmation.
Timeframe: 4h
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
    
    # Calculate TRIX (15-period EMA of EMA of EMA of ROC)
    # TRIX = EMA(EMA(EMA(ROC, 15), 15), 15)
    roc = np.diff(np.log(close), prepend=np.log(close[0])) * 100  # approximate ROC %
    
    # Three-fold EMA smoothing
    ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = ema3
    
    # Volume confirmation (20-period average)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14-period) for regime detection
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]  # first TR
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    
    max_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range14 = max_high14 - min_low14
    
    # Avoid division by zero
    chop = np.full_like(close, 50.0)  # default to neutral
    mask = (range14 > 0) & (~np.isnan(range14))
    chop[mask] = 100 * np.log10(sum_atr14[mask] / range14[mask]) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(15, 20, 14)  # max of TRIX, volume MA, CHOP periods
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(trix[i]) or np.isnan(volume_ma20[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Regime filter: CHOP > 61.8 indicates ranging market (good for mean reversion)
        ranging_filter = chop[i] > 61.8
        
        if position == 0:
            # Long: TRIX positive + volume spike + ranging market
            if trix[i] > 0 and volume_filter and ranging_filter:
                signals[i] = 0.25
                position = 1
            # Short: TRIX negative + volume spike + ranging market
            elif trix[i] < 0 and volume_filter and ranging_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX turns negative OR market breaks out of range (CHOP < 38.2)
            if trix[i] <= 0 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX turns positive OR market breaks out of range (CHOP < 38.2)
            if trix[i] >= 0 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TRIX_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0