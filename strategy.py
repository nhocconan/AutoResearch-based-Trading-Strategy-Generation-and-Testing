#!/usr/bin/env python3
"""
12h_TRIX_VolumeSpike_ChopRegime_V1
Hypothesis: TRIX momentum with volume confirmation and choppiness regime filter works on 12h timeframe for BTC and ETH in both bull and bear markets. Uses 1d timeframe for TRIX calculation and volume spike confirmation. Choppiness index filters for trending markets (CHOP < 38.2). Target: 12-37 trades/year per symbol (50-150 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data once for TRIX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate TRIX (15-period EMA applied 3 times)
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix[0] = np.nan
    
    # Align TRIX to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Volume filter: 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(atr_sum / (hh - ll)) / log10(14)
    hh_ll = hh - ll
    chop = np.full_like(close, np.nan, dtype=float)
    mask = (hh_ll > 0) & (~np.isnan(atr_sum))
    chop[mask] = 100 * np.log10(atr_sum[mask] / hh_ll[mask]) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation
        volume_ok = volume > 1.5 * vol_ma[i]
        
        # Choppiness regime filter (trending market)
        trending = chop[i] < 38.2
        
        if position == 0:
            # Long: TRIX crosses above zero with volume and trending regime
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and volume_ok and trending:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume and trending regime
            elif trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and volume_ok and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: TRIX crosses below zero or opposite signal
            if trix_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: TRIX crosses above zero or opposite signal
            if trix_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_TRIX_VolumeSpike_ChopRegime_V1"
timeframe = "12h"
leverage = 1.0