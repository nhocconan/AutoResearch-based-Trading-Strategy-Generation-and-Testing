#!/usr/bin/env python3
# 4h_1d_Trix_ZeroCross_VolumeTrend
# Hypothesis: 4h TRIX zero-cross signals filtered by 1d EMA50 trend and volume surge.
# TRIX (triple-smoothed EMA) captures momentum with less noise; zero-cross indicates trend change.
# Combined with 1d trend alignment and volume confirmation for high-probability entries.
# Designed for low trade frequency (<50/year) to minimize fee drag and work in bull/bear markets.

name = "4h_1d_Trix_ZeroCross_VolumeTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # TRIX on 4h: triple EMA of close, then percent change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3.pct_change())
    trix_values = trix.values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need TRIX (12*3=36) + volume MA (20) + EMA (50)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(trix_values[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1d EMA50
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        uptrend = close_1d_aligned[i] > ema_50_1d_aligned[i]
        downtrend = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        # TRIX zero-cross signals
        trix_cross_up = trix_values[i] > 0 and trix_values[i-1] <= 0
        trix_cross_down = trix_values[i] < 0 and trix_values[i-1] >= 0
        
        if position == 0:
            # Long: TRIX crosses above zero with volume surge and 1d uptrend
            if trix_cross_up and volume_surge and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume surge and 1d downtrend
            elif trix_cross_down and volume_surge and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero OR trend changes
            if trix_cross_down or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero OR trend changes
            if trix_cross_up or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals