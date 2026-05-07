#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_TrendFilter_v1
Hypothesis: Uses TRIX (15-period) for momentum with volume spike confirmation and EMA50 trend filter. 
Enters long when TRIX crosses above zero with volume > 2x average and price above EMA50. 
Enters short when TRIX crosses below zero with volume > 2x average and price below EMA50. 
Designed for low trade frequency (20-40/year) with clear momentum logic, works in trending markets and avoids range-bound conditions.
"""

name = "4h_TRIX_VolumeSpike_TrendFilter_v1"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # TRIX calculation (15-period EMA of EMA of EMA)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix[0] = 0  # First value undefined
    
    # Volume confirmation: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine daily trend using aligned close
        daily_close_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
        if np.isnan(daily_close_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        daily_trend_up = daily_close_aligned[i] > ema_50_1d_aligned[i]
        daily_trend_down = daily_close_aligned[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: TRIX crosses above zero, volume spike, price above EMA50
            if (trix[i] > 0 and trix[i-1] <= 0 and 
                vol_ratio[i] > 2.0 and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero, volume spike, price below EMA50
            elif (trix[i] < 0 and trix[i-1] >= 0 and 
                  vol_ratio[i] > 2.0 and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX crosses below zero or price falls below EMA50
            if trix[i] < 0 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX crosses above zero or price rises above EMA50
            if trix[i] > 0 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals