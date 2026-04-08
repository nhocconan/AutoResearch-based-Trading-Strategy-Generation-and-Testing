#!/usr/bin/env python3
# 4h_trix_volume_regime_v1
# Hypothesis: TRIX momentum combined with volume confirmation and choppiness regime filter on 4h.
# Long when TRIX crosses above zero with volume > 1.5x average and choppiness > 61.8 (ranging market).
# Short when TRIX crosses below zero with volume > 1.5x average and choppiness > 61.8.
# Exit when TRIX crosses back through zero.
# Designed to capture momentum reversals in ranging markets with volume confirmation.
# Target: 80-160 total trades over 4 years (~20-40/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_trix_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate TRIX (12-period EMA of EMA of EMA of ROC)
    # ROC = (close - close.shift(1)) / close.shift(1)
    close_series = pd.Series(close)
    roc = close_series.pct_change(1)
    ema1 = roc.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ema3.values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR(1)) / (n * (highest(high) - lowest(low)))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First TR has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (14 * (highest_high - lowest_low))) / np.log10(14)
    chop = chop.values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 14
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or 
            np.isnan(avg_volume[i]) or np.isnan(chop[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: TRIX crosses below zero
            if trix[i] < 0 and trix[i-1] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TRIX crosses above zero
            if trix[i] > 0 and trix[i-1] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            # Regime filter: choppiness > 61.8 (ranging market)
            chop_ok = chop[i] > 61.8
            
            # TRIX zero-cross entries
            if (trix[i] > 0 and trix[i-1] <= 0) and volume_ok and chop_ok:
                position = 1
                signals[i] = 0.25
            elif (trix[i] < 0 and trix[i-1] >= 0) and volume_ok and chop_ok:
                position = -1
                signals[i] = -0.25
    
    return signals