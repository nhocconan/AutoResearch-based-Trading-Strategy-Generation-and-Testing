#!/usr/bin/env python3
# 1D_TRIX_Threshold_1wTrend_VolumeFilter
# Hypothesis: TRIX zero-cross signals filtered by 1w trend and volume spikes.
# Long when: TRIX crosses above zero, 1w trend up, volume > 1.5x average.
# Short when: TRIX crosses below zero, 1w trend down, volume > 1.5x average.
# Works in bull/bear by following weekly trend and using volume to confirm institutional interest.
# Target: 10-25 trades/year per symbol.

name = "1D_TRIX_Threshold_1wTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0

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
    
    # TRIX calculation (15-period EMA of EMA of EMA)
    close_s = pd.Series(close)
    ema1 = close_s.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = 100 * (ema3.pct_change())
    
    # TRIX previous value for zero-cross detection
    trix_prev = trix.shift(1)
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > ema50_1w
    weekly_downtrend = close_1w < ema50_1w
    
    # Align weekly trend to daily
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    # Volume average (20-period)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix[i]) or np.isnan(trix_prev[i]) or np.isnan(vol_ma[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        weekly_up = weekly_uptrend_aligned[i] > 0.5
        weekly_down = weekly_downtrend_aligned[i] > 0.5
        
        # TRIX zero-cross detection
        trix_cross_up = trix[i] > 0 and trix_prev[i] <= 0
        trix_cross_down = trix[i] < 0 and trix_prev[i] >= 0
        
        if position == 0:
            # Enter long: TRIX crosses up + weekly uptrend + volume
            if trix_cross_up and weekly_up and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses down + weekly downtrend + volume
            elif trix_cross_down and weekly_down and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: TRIX crosses down or weekly trend changes
            if trix_cross_down or not weekly_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: TRIX crosses up or weekly trend changes
            if trix_cross_up or not weekly_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals