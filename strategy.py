#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_Regime_v2
TRIX(12) crossing zero line as momentum signal.
Volume spike > 2x 20-period average for confirmation.
Choppiness regime filter: CHOP(14) < 38.2 = trending (follow TRIX), CHOP > 61.8 = range (avoid).
Exit when TRIX crosses back or volume drops below average.
Designed to capture momentum bursts in trending regimes with volume confirmation.
Target: 80-160 total trades over 4 years (20-40/year).
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
    # Triple EMA of closing price
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # TRIX = 100 * (ema3_today - ema3_yesterday) / ema3_yesterday
    trix = np.zeros_like(close)
    trix[1:] = 100 * (ema3[1:] - ema3[:-1]) / (ema3[:-1] + 1e-10)
    
    # === Volume Spike (2x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    # === Choppiness Index (14) ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(tr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero
    hh_ll = hh - ll
    hh_ll = np.where(hh_ll == 0, 1e-10, hh_ll)
    chop = 100 * np.log10(tr_sum / hh_ll) / np.log10(14)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmrow = 30
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmrow, n):
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: TRIX crosses above zero, volume spike, trending regime (CHOP < 38.2)
            if (trix[i] > 0 and trix[i-1] <= 0 and 
                vol_spike[i] and 
                chop[i] < 38.2):
                signals[i] = 0.25
                position = 1
                continue
            # Short: TRIX crosses below zero, volume spike, trending regime (CHOP < 38.2)
            elif (trix[i] < 0 and trix[i-1] >= 0 and 
                  vol_spike[i] and 
                  chop[i] < 38.2):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: TRIX crosses below zero OR volume drops below average OR ranging regime (CHOP > 61.8)
            if (trix[i] < 0 and trix[i-1] >= 0 or 
                volume[i] < vol_ma[i] or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero OR volume drops below average OR ranging regime (CHOP > 61.8)
            if (trix[i] > 0 and trix[i-1] <= 0 or 
                volume[i] < vol_ma[i] or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TRIX_VolumeSpike_Regime_v2"
timeframe = "4h"
leverage = 1.0