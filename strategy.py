#!/usr/bin/env python3
# 6H_1D_Trix_20_VolumeSpike_1dTrend
# Hypothesis: Use TRIX(20) momentum on 6h for entry timing, with 1d EMA34 as trend filter.
# Volume confirmation (>1.5x 20-period average) ensures breakouts have conviction.
# TRIX helps capture momentum shifts, effective in both bull and bear markets via trend filter.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6H_1D_Trix_20_VolumeSpike_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate TRIX (20) on 6h close
    # TRIX = EMA(EMA(EMA(close, 20), 20), 20) - 1-period percent change
    ema1 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema2 = pd.Series(ema1).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema3 = pd.Series(ema2).ewm(span=20, adjust=False, min_periods=20).mean().values
    trix = np.zeros_like(close)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100  # percent change
    
    # Calculate daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 60  # enough for TRIX calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(trix[i]) or np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: TRIX crosses above zero + above daily EMA34 + volume confirmation
            if trix[i] > 0 and trix[i-1] <= 0 and close[i] > ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below zero + below daily EMA34 + volume confirmation
            elif trix[i] < 0 and trix[i-1] >= 0 and close[i] < ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below zero or below daily EMA34
            if trix[i] < 0 and trix[i-1] >= 0 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero or above daily EMA34
            if trix[i] > 0 and trix[i-1] <= 0 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals