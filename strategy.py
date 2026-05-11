#!/usr/bin/env python3
"""
4h_Trix_Volume_Spike_1dTrend
Hypothesis: TRIX (triple exponential average) crossing zero with volume spike and 1-day trend filter.
Works in both bull and bear markets: long when TRIX crosses above zero in uptrend (close > 1d EMA),
short when TRIX crosses below zero in downtrend (close < 1d EMA). Volume spike confirms momentum.
Designed for low trade frequency (<30/year) on 4h to minimize fee drag.
"""

name = "4h_Trix_Volume_Spike_1dTrend"
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
    
    # === Get 1d data for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # === TRIX calculation (15-period) ===
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - 1-period percent change
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1) * 100
    trix[0] = 0  # first value has no previous
    
    # === 1-day EMA34 Trend Filter ===
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Volume Spike Filter (2x 20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 2.0
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers TRIX and EMA calculation)
    start_idx = 45
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(trix[i]) or np.isnan(ema34_1d_4h[i]) or 
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero with uptrend (close > EMA34) and volume spike
            if (trix[i] > 0 and trix[i-1] <= 0 and 
                close[i] > ema34_1d_4h[i] and volume_ok[i]):
                signals[i] = position_size
                position = 1
            # Short: TRIX crosses below zero with downtrend (close < EMA34) and volume spike
            elif (trix[i] < 0 and trix[i-1] >= 0 and 
                  close[i] < ema34_1d_4h[i] and volume_ok[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit: TRIX crosses back through zero in opposite direction
            if position == 1:
                if trix[i] < 0:  # Exit long if TRIX turns negative
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if trix[i] > 0:  # Exit short if TRIX turns positive
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals