#!/usr/bin/env python3
"""
4h_TRIX_ZeroLag_VolumeSpike_Regime
Hypothesis: 4h TRIX zero-cross with volume spike confirmation and choppiness regime filter.
Long when TRIX crosses above zero with volume spike and trending regime (CHOP < 38.2).
Short when TRIX crosses below zero with volume spike and trending regime (CHOP < 38.2).
Uses zero-lag TRIX to reduce lag and improve signal timing.
Targets 20-40 trades/year on 4h timeframe to minimize fee drag while capturing momentum shifts.
Works in both bull and bear markets by following momentum direction with regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Zero-lag TRIX calculation
    # EMA1 = EMA(close, 12)
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA2 = EMA(EMA1, 12)
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA3 = EMA(EMA2, 12)
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # TRIX = (EMA3 - prev_EMA3) / prev_EMA3 * 100
    trix_raw = np.zeros_like(close)
    trix_raw[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    # Zero-lag adjustment: TRIX + (TRIX - delayed_TRIX)
    trix_delayed = np.roll(trix_raw, 1)
    trix_delayed[0] = 0
    trix = trix_raw + (trix_raw - trix_delayed)
    
    # Choppiness Index regime filter (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    
    # Sum of True Range over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Chop = 100 * log10(atr_sum / (hh - ll)) / log10(14)
    chop = np.zeros_like(close)
    denominator = hh - ll
    mask = (denominator > 0) & ~np.isnan(denominator)
    chop[mask] = 100 * np.log10(atr_sum[mask] / denominator[mask]) / np.log10(14)
    # Trending regime: CHOP < 38.2
    trending_regime = chop < 38.2
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for TRIX (~36), Chop (~14), Volume MA (~20)
    start_idx = max(36, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero + volume spike + trending regime
            long_setup = (trix[i] > 0) and (trix[i-1] <= 0) and volume_spike[i] and trending_regime[i]
            # Short: TRIX crosses below zero + volume spike + trending regime
            short_setup = (trix[i] < 0) and (trix[i-1] >= 0) and volume_spike[i] and trending_regime[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: TRIX crosses below zero OR regime changes to choppy
            if (trix[i] < 0 and trix[i-1] >= 0) or (chop[i] >= 38.2):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: TRIX crosses above zero OR regime changes to choppy
            if (trix[i] > 0 and trix[i-1] <= 0) or (chop[i] >= 38.2):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_TRIX_ZeroLag_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0