#!/usr/bin/env python3
"""
4h_TRIX_ZeroCross_VolumeSpike_ChopRegime_v1
Hypothesis: 4h TRIX zero-cross with volume spike and choppiness regime filter.
- TRIX (15-period) crossing above/below zero-line as momentum signal
- Volume confirmation: current volume > 1.5x 20-period average
- Choppiness regime: CHOP(14) < 38.2 = trending (favor TRIX signals), CHOP > 61.8 = range (avoid)
- Designed for low trade frequency with edge in both bull and bear markets via momentum + regime
- Target: 75-200 total trades over 4 years (19-50/year) on BTC/ETH/SOL
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === TRIX calculation (15-period) ===
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - 1 period ago
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1) * 100
    trix[0] = 0  # First value has no previous
    
    # === Choppiness Index (14-period) ===
    # CHOP = 100 * log10(sum(ATR(1), 14) / (log10(highest_high - lowest_low) * log10(14)))
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]  # First TR
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_hl = highest_high - lowest_low
    chop = np.zeros_like(close)
    mask = (range_hl > 0) & ~np.isnan(range_hl)
    chop[mask] = 100 * np.log10(sum_atr1[mask] / (np.log10(range_hl[mask]) * np.log10(14)))
    chop[~mask] = 50  # Default middle value when range is zero
    
    # === Volume spike (20-period average) ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 45 for TRIX, 14 for CHOP, 20 for volume)
    start_idx = max(45, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix[i]) or np.isnan(chop[i]) or np.isnan(vol_ma20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # TRIX zero-cross signals
        trix_cross_up = trix[i] > 0 and trix[i-1] <= 0
        trix_cross_down = trix[i] < 0 and trix[i-1] >= 0
        
        # Choppiness regime: favor trending markets (CHOP < 38.2), avoid ranging (CHOP > 61.8)
        chop_regime = chop[i] < 38.2  # True = trending, False = ranging/choppy
        
        if position == 0:
            # Long: TRIX crosses above zero AND volume spike AND trending regime
            if trix_cross_up and volume_spike[i] and chop_regime:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero AND volume spike AND trending regime
            elif trix_cross_down and volume_spike[i] and chop_regime:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: TRIX crosses below zero OR chop becomes too high (ranging)
            if trix_cross_down or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TRIX crosses above zero OR chop becomes too high (ranging)
            if trix_cross_up or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_TRIX_ZeroCross_VolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0