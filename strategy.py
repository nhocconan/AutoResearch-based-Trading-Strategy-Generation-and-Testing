#!/usr/bin/env python3
"""
12H_TRIX_Momentum_1DTrend_VolumeFilter
Hypothesis: 12h TRIX momentum crossovers with 1D EMA34 trend confirmation and volume filter.
TRIX filters noise and captures momentum shifts; EMA34 ensures alignment with daily trend.
Volume filter confirms breakout strength. Targets 12-37 trades/year to minimize fee drag on 12h timeframe.
Works in bull/bear markets: TRIX captures momentum reversals, EMA34 filters counter-trend moves.
"""
name = "12H_TRIX_Momentum_1DTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1D data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1D EMA34 for trend direction
    close_1d_series = pd.Series(df_1d['close'])
    ema_34 = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate TRIX on 12h close (15-period EMA of 15-period EMA of 15-period EMA)
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 100
    trix_values = trix.values
    
    # Volume filter: current 12h volume > 1.5 x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(45, 20)  # Ensure sufficient warmup for TRIX (3*15=45) and volume
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(trix_values[i]) or 
            np.isnan(trix_values[i-1]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 48 bars between trades (8 days on 12h TF) to reduce frequency
            if bars_since_exit < 48:
                continue
                
            # Long: TRIX crosses above zero with EMA34 uptrend and volume spike
            if (trix_values[i] > 0 and trix_values[i-1] <= 0 and 
                close[i] > ema_34_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: TRIX crosses below zero with EMA34 downtrend and volume spike
            elif (trix_values[i] < 0 and trix_values[i-1] >= 0 and 
                  close[i] < ema_34_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: TRIX crosses back through zero (momentum reversal)
            if position == 1 and trix_values[i] < 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and trix_values[i] > 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals