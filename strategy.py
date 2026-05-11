#!/usr/bin/env python3
name = "1d_Trix_1wTrend_VolumeFilter"
timeframe = "1d"
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
    
    # TRIX: 15-period EMA applied 3 times on close
    ema1 = pd.Series(close).ewm(span=15, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False).mean().values
    trix = np.diff(ema3, prepend=ema3[0]) / ema3
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current volume > 1.3x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.3 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Enough for TRIX and 1w EMA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(trix[i]) or np.isnan(ema50_1w_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero + price above 1w EMA50 + volume
            if trix[i] > 0 and trix[i-1] <= 0 and close[i] > ema50_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero + price below 1w EMA50 + volume
            elif trix[i] < 0 and trix[i-1] >= 0 and close[i] < ema50_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero OR price crosses below 1w EMA50
            if trix[i] < 0 and trix[i-1] >= 0 or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero OR price crosses above 1w EMA50
            if trix[i] > 0 and trix[i-1] <= 0 or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals