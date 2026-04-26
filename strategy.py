#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_ChopRegime
Hypothesis: TRIX (15-period) crossing zero line with volume spike (>2.0x 20-bar MA) and choppiness regime filter (CHOP < 38.2 = trending). TRIX is a momentum oscillator that filters out insignificant cycles and is effective in both bull and bear markets. Volume confirmation ensures breakout strength, while chop filter avoids range-bound whipsaws. Target: 20-40 trades/year.
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
    
    # TRIX calculation (15-period, 3x EMA smoothing)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = np.where(ema3[:-1] != 0, (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100, 0)
    trix = np.concatenate([np.full(15, np.nan), trix])  # Align length
    
    # Choppiness regime filter: CHOP(14) > 61.8 = range, CHOP < 38.2 = trending
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((atr_14 * 14) / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop_regime = (chop < 38.2)  # Only trade in trending regime
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size (25% of capital)
    
    # Warmup: max of calculations (15 for TRIX, 20 for volume, 14 for chop)
    start_idx = max(15, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(chop[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        trix_val = trix[i]
        trix_prev = trix[i-1]
        vol_spike = volume_spike[i]
        in_trend = chop_regime[i]
        
        # Entry conditions: TRIX crossing zero line with volume spike and in trending regime
        long_entry = (trix_prev <= 0 and trix_val > 0) and vol_spike and in_trend
        short_entry = (trix_prev >= 0 and trix_val < 0) and vol_spike and in_trend
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
            elif short_entry:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when TRIX crosses below zero or chop regime shifts
            if trix_val < 0 or not in_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = base_size
        elif position == -1:
            # Short - exit when TRIX crosses above zero or chop regime shifts
            if trix_val > 0 or not in_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0