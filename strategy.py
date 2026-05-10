#!/usr/bin/env python3
# 4h_TRIX_15_VolumeSpike_ChoppyExit
# Hypothesis: Uses TRIX(15) momentum with volume spike entry and choppy market exit.
# Long when TRIX crosses above zero with volume > 2x average and market not choppy.
# Short when TRIX crosses below zero with volume > 2x average and market not choppy.
# Exit when market becomes choppy (Choppiness Index > 61.8) or TRIX reverses.
# Designed for ~30-50 trades/year to avoid overtrading and work in trending markets.

name = "4h_TRIX_15_VolumeSpike_ChoppyExit"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate TRIX(15) - triple smoothed EMA
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix[0] = 0  # First value has no previous
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Choppiness Index (14 periods) for regime filter
    def true_range(h, l, c_prev):
        return np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    
    tr = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr14 = np.full(n, np.nan)
    for i in range(14, n):
        atr14[i] = np.mean(tr[i-13:i+1])
    
    chop = np.full(n, np.nan)
    for i in range(14, n):
        if atr14[i] > 0:
            sum_tr14 = np.sum(tr[i-13:i+1])
            max_h = np.max(high[i-13:i+1])
            min_l = np.min(low[i-13:i+1])
            chop[i] = 100 * np.log10(sum_tr14 / (max_h - min_l)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # TRIX needs 30, vol needs 20
    
    for i in range(start_idx, n):
        if np.isnan(trix[i]) or np.isnan(vol_ma[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero with volume confirmation and not choppy
            if trix[i] > 0 and trix[i-1] <= 0 and volume[i] > 2.0 * vol_ma[i] and chop[i] <= 61.8:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume confirmation and not choppy
            elif trix[i] < 0 and trix[i-1] >= 0 and volume[i] > 2.0 * vol_ma[i] and chop[i] <= 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: TRIX crosses below zero OR market becomes choppy
            if trix[i] < 0 and trix[i-1] >= 0 or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: TRIX crosses above zero OR market becomes choppy
            if trix[i] > 0 and trix[i-1] <= 0 or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals