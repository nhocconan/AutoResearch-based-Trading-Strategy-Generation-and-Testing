#!/usr/bin/env python3
"""
4h_TRIX_ZeroCross_VolumeSpike_TrendFilter
Hypothesis: TRIX zero-cross with volume spike and 1d EMA trend filter on 4h timeframe.
Goes long when TRIX crosses above zero with volume spike and uptrend (price > EMA34).
Goes short when TRIX crosses below zero with volume spike and downtrend (price < EMA34).
Designed for low trade frequency (20-50/year) to avoid fee decay while capturing
momentum in both bull and bear markets via trend alignment and momentum confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # TRIX calculation (15-period)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix[0] = 0  # First value has no previous
    
    # TRIX zero-cross signals
    trix_cross_up = (trix > 0) & (np.roll(trix, 1) <= 0)
    trix_cross_down = (trix < 0) & (np.roll(trix, 1) >= 0)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 20)  # Warmup for TRIX and volume
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema34 = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: TRIX crosses above zero with volume spike and uptrend
            if trix_cross_up[i] and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume spike and downtrend
            elif trix_cross_down[i] and vol_spike and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: TRIX crosses below zero OR trend turns down
            if trix_cross_down[i] or price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: TRIX crosses above zero OR trend turns up
            if trix_cross_up[i] or price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_TRIX_ZeroCross_VolumeSpike_TrendFilter"
timeframe = "4h"
leverage = 1.0