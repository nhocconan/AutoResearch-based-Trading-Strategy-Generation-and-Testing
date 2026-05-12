#!/usr/bin/env python3
name = "1d_Trix_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def trix(close, period=15):
    """TRIX: triple smoothed EMA rate of change."""
    ema1 = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean()
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
    ema3 = ema2.ewm(span=period, adjust=False, min_periods=period).mean()
    trix_val = ema3.pct_change() * 100
    return trix_val.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly TRIX for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    trix_1w = trix(close_1w, period=15)
    trix_1w_aligned = align_htf_to_ltf(prices, df_1w, trix_1w)
    
    # Daily TRIX for signal
    trix_1d = trix(close, period=15)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure TRIX has enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_1w_aligned[i]) or 
            np.isnan(trix_1d[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero + weekly TRIX positive + volume filter
            if trix_1d[i] > 0 and trix_1d[i-1] <= 0 and trix_1w_aligned[i] > 0 and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero + weekly TRIX negative + volume filter
            elif trix_1d[i] < 0 and trix_1d[i-1] >= 0 and trix_1w_aligned[i] < 0 and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX crosses below zero
            if trix_1d[i] < 0 and trix_1d[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX crosses above zero
            if trix_1d[i] > 0 and trix_1d[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals