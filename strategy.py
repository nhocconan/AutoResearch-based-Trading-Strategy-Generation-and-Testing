#!/usr/bin/env python3
"""
4h_TRIX_ZeroCross_VolumeSpike_TrendFilter
Hypothesis: TRIX (1-period rate of change of triple-smoothed EMA) zero cross with volume spike and 4h EMA50 trend filter.
TRIX captures momentum shifts early; zero cross indicates trend change. Volume confirms breakout strength.
Works in both bull/bear markets by using EMA50 for trend direction and requiring volume spike to avoid false signals.
Target: 20-35 trades/year to avoid fee drag.
"""

name = "4h_TRIX_ZeroCross_VolumeSpike_TrendFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data for TRIX calculation (using same timeframe as primary)
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 1:
        return np.zeros(n)
    
    # Calculate TRIX: 1-period ROC of triple-smoothed EMA (15-period)
    close = df_4h['close'].values
    # Triple EMA: EMA(EMA(EMA(close, 15), 15), 15)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # TRIX = 100 * (ema3 - ema3_prev) / ema3_prev
    trix_raw = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix_raw[0] = 0  # first value undefined
    
    # Align TRIX to 4h timeframe (same as primary, so no shift needed but use for consistency)
    trix = align_htf_to_ltf(prices, df_4h, trix_raw)
    
    # Get EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Get price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 2.0x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need TRIX (15*3=45) and EMA50 (50)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(trix[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero with uptrend and volume
            if trix[i] > 0 and trix[i-1] <= 0 and close[i] > ema_50_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with downtrend and volume
            elif trix[i] < 0 and trix[i-1] >= 0 and close[i] < ema_50_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero or trend change
            if trix[i] < 0 and trix[i-1] >= 0 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero or trend change
            if trix[i] > 0 and trix[i-1] <= 0 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals