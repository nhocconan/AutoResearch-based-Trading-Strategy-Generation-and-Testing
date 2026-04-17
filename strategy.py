#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_Regime_v1
TRIX(12) > 0 + Volume Spike + Chop Regime Filter (CHOP > 61.8 = range)
Long when TRIX crosses above zero in ranging market with volume confirmation.
Short when TRIX crosses below zero in ranging market with volume confirmation.
Exit when TRIX crosses back to zero or chop regime ends.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === TRIX(12) ===
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix[0] = 0  # first value
    
    # === Volume Spike (2x 20-period average) ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma20)
    
    # === Chopiness Index (14) ===
    atr1 = np.abs(high - low)
    atr2 = np.abs(high - np.roll(close, 1))
    atr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(atr1, np.maximum(atr2, atr3))
    tr[0] = atr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)  # avoid div0
    
    signals = np.zeros(n)
    
    # Warmup period
    warmrow = 30
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmrow, n):
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: TRIX crosses above zero, chop > 61.8 (range), volume spike
            if (trix[i] > 0 and trix[i-1] <= 0 and 
                chop[i] > 61.8 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: TRIX crosses below zero, chop > 61.8 (range), volume spike
            elif (trix[i] < 0 and trix[i-1] >= 0 and 
                  chop[i] > 61.8 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: TRIX crosses below zero OR chop < 38.2 (trending)
            if (trix[i] < 0 and trix[i-1] >= 0) or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero OR chop < 38.2 (trending)
            if (trix[i] > 0 and trix[i-1] <= 0) or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TRIX_VolumeSpike_Regime_v1"
timeframe = "4h"
leverage = 1.0