#!/usr/bin/env python3
"""
12h_TRIX_VolumeSpike_TrendFilter
Hypothesis: Use TRIX (15-period) for momentum, volume spike (>2x 20-period average) for confirmation, and 1d EMA50 for trend filter. Go long when TRIX crosses above zero with volume confirmation and price above EMA50, short when TRIX crosses below zero with volume confirmation and price below EMA50. Exit on opposite TRIX cross. Designed for 12h timeframe to keep trades low (12-37/year) and avoid fee drift. Works in bull (momentum with trend) and bear (mean reversion via TRIX zero-cross in ranging markets with trend filter).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate TRIX from close prices
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix_values = trix.values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # Get 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for TRIX (45), volume average (20), and EMA (50)
    start_idx = max(45, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix_values[i]) or np.isnan(trix_values[i-1]) or 
            np.isnan(volume_confirm[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        trix_now = trix_values[i]
        trix_prev = trix_values[i-1]
        vol_conf = volume_confirm[i]
        ema_50_val = ema_50_aligned[i]
        
        if position == 0:
            # Long: TRIX crosses above zero with volume confirmation AND price above EMA50 (uptrend)
            if trix_prev <= 0 and trix_now > 0 and vol_conf and close[i] > ema_50_val:
                signals[i] = size
                position = 1
            # Short: TRIX crosses below zero with volume confirmation AND price below EMA50 (downtrend)
            elif trix_prev >= 0 and trix_now < 0 and vol_conf and close[i] < ema_50_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: TRIX crosses below zero
            if trix_prev >= 0 and trix_now < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: TRIX crosses above zero
            if trix_prev <= 0 and trix_now > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_TRIX_VolumeSpike_TrendFilter"
timeframe = "12h"
leverage = 1.0