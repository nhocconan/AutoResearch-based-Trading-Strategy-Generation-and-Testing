#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX momentum with volume confirmation and daily trend filter.
# TRIX (triple-smoothed EMA) identifies momentum with less noise.
# Long when TRIX crosses above zero AND 1d EMA34 rising AND volume > 2x 20-period average.
# Short when TRIX crosses below zero AND 1d EMA34 falling AND volume > 2x 20-period average.
# Exit when TRIX crosses back to zero (momentum reversal).
# TRIX is effective in both trending and ranging markets, providing early momentum signals.
# Volume confirmation ensures institutional participation. Daily trend filter aligns with higher timeframe bias.
# Target: 25-40 trades/year (100-160 total over 4 years) to minimize fee drag.
# Works in bull markets via momentum continuation and in bear markets via counter-trend reversals.

name = "4h_TRIX_ZeroCross_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate TRIX (15-period triple EMA of ROC)
    # ROC = (close / close.shift(1) - 1) * 100
    close_series = pd.Series(close)
    roc = (close_series / close_series.shift(1) - 1) * 100
    # Triple EMA of ROC
    ema1 = roc.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = ema3.values
    
    # Align daily EMA34 to 4h timeframe
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily EMA34 direction
    ema34_rising = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_falling = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_rising[1:] = ema34_1d_aligned[1:] > ema34_1d_aligned[:-1]
    ema34_falling[1:] = ema34_1d_aligned[1:] < ema34_1d_aligned[:-1]
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(45, 20)  # Sufficient warmup for TRIX (3*15=45) and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_rising[i]) or 
            np.isnan(ema34_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: TRIX crosses above zero, 1d EMA34 rising, volume filter
            long_cond = (trix[i] > 0) and (trix[i-1] <= 0) and ema34_rising[i] and volume_filter[i]
            # Short conditions: TRIX crosses below zero, 1d EMA34 falling, volume filter
            short_cond = (trix[i] < 0) and (trix[i-1] >= 0) and ema34_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses back below zero
            if trix[i] < 0 and trix[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses back above zero
            if trix[i] > 0 and trix[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals