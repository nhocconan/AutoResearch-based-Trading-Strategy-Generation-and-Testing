#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_1dTrend
Hypothesis: TRIX momentum combined with volume spike and 1d trend filter to capture breakout moves in both bull and bear markets.
Long when TRIX crosses above zero + volume spike + price > 1d EMA50.
Short when TRIX crosses below zero + volume spike + price < 1d EMA50.
Uses discrete position sizing to limit trade frequency and control drawdown.
"""

name = "4h_TRIX_VolumeSpike_1dTrend"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1D Data for Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Trend filter: EMA50 on 1d close
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # TRIX: 1-period ROC of triple-smoothed EMA(15)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = ema3.pct_change() * 100  # percentage change
    trix_values = trix.values
    
    # Volume spike: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for TRIX and EMA
    start_idx = 45
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(trix_values[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero + volume spike + uptrend
            if trix_values[i] > 0 and trix_values[i-1] <= 0 and volume_spike[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero + volume spike + downtrend
            elif trix_values[i] < 0 and trix_values[i-1] >= 0 and volume_spike[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero OR price breaks below EMA50
            if trix_values[i] < 0 and trix_values[i-1] >= 0 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: TRIX crosses above zero OR price breaks above EMA50
            if trix_values[i] > 0 and trix_values[i-1] <= 0 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals