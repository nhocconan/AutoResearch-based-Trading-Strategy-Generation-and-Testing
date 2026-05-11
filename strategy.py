#!/usr/bin/env python3
name = "1d_TRIX_ZeroCross_1wTrend_Volume"
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
    
    # Get 1w data for weekly trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_1w = close_1w > ema50_1w
    
    # Calculate TRIX on daily close
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix_raw = (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1) * 100
    trix_raw[0] = 0  # First value has no previous
    
    # TRIX smoothed (signal line)
    trix_signal = pd.Series(trix_raw).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # TRIX histogram (main signal)
    trix_hist = trix_raw - trix_signal
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 1d timeframe
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    trix_hist_aligned = align_htf_to_ltf(prices, df_1d, trix_hist)
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(trix_hist_aligned[i]) or 
            np.isnan(trend_up_1w_aligned[i]) or
            np.isnan(vol_ma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero + weekly uptrend + volume confirmation
            if (trix_hist_aligned[i] > 0 and trix_hist_aligned[i-1] <= 0 and 
                trend_up_1w_aligned[i] and 
                volume[i] > 1.3 * vol_ma20_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero + weekly downtrend + volume confirmation
            elif (trix_hist_aligned[i] < 0 and trix_hist_aligned[i-1] >= 0 and 
                  not trend_up_1w_aligned[i] and 
                  volume[i] > 1.3 * vol_ma20_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero or trend changes
            if (trix_hist_aligned[i] < 0 or 
                not trend_up_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero or trend changes
            if (trix_hist_aligned[i] > 0 or 
                trend_up_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals