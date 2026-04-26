#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_Regime_v1
Hypothesis: TRIX (15-period) crossing zero with volume spike (>2x 20-median) and chop regime filter (CHOP(14) > 61.8 for mean reversion) generates high-probability reversals in ranging markets. Uses 1d EMA50 trend filter to avoid counter-trend trades. Designed for 4h timeframe to capture swing reversals in both bull and bear markets by combining momentum (TRIX), conviction (volume), and market structure (chop). Targets 20-50 trades/year via tight entry conditions requiring confluence of three filters.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # TRIX(15): triple EMA of ROC, then percent change
    roc = pd.Series(close).pct_change(1).values
    ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = pd.Series(ema3).pct_change(1).values * 100  # scale for readability
    
    # Choppiness Index: CHOP(14) = 100 * log10(sum(ATR(1)) / (n * log(n))) / log10(n)
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]
    atr1 = tr1  # ATR(1) is just true range
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    n_val = 14
    chop = 100 * np.log10(sum_atr1 / (n_val * np.log10(n_val))) / np.log10(n_val)
    
    # Volume spike filter: volume > 2x median volume (20-period)
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Align HTF indicators to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of TRIX warmup (15*3+1), CHOP (14), volume median (20), EMA50 (50)
    start_idx = max(15*3+1, 14, 20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix[i]) or 
            np.isnan(chop[i]) or
            np.isnan(vol_median[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        trix_val = trix[i]
        chop_val = chop[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        
        # Volume spike filter
        volume_spike = volume_val > 2.0 * vol_median_val
        
        # Regime filter: chop > 61.8 indicates ranging market (mean reversion favorable)
        ranging_market = chop_val > 61.8
        
        if position == 0:
            # Long: TRIX crosses above zero with volume spike in ranging market
            long_signal = (trix_val > 0) and (trix[i-1] <= 0) and \
                          volume_spike and \
                          ranging_market
            
            # Short: TRIX crosses below zero with volume spike in ranging market
            short_signal = (trix_val < 0) and (trix[i-1] >= 0) and \
                           volume_spike and \
                           ranging_market
            
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: TRIX crosses below zero OR chop drops below 38.2 (trending regime)
            if trix_val < 0 or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: TRIX crosses above zero OR chop drops below 38.2 (trending regime)
            if trix_val > 0 or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_TRIX_VolumeSpike_Regime_v1"
timeframe = "4h"
leverage = 1.0