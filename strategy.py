#!/usr/bin/env python3
# 1d_1w_WeeklyPivot_TrendFollowing
# Hypothesis: Follow the weekly trend using 1w EMA21 as trend filter, and enter on pullbacks to 1d 20-period EMA.
# In bull markets, price stays above weekly EMA21; in bear markets, stays below. Pullbacks to daily EMA offer high-probability entries.
# Uses volume confirmation (>1.5x 20-day average) to filter weak moves.
# Target: 30-100 total trades over 4 years (7-25/year) with discrete sizing to minimize fee drag.

name = "1d_1w_WeeklyPivot_TrendFollowing"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate weekly EMA21 for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate daily EMA20 for entry trigger
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume average for spike detection (20-day)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(ema_20[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly EMA21 and pulling back to daily EMA20 with volume
            if (close[i] > ema_21_1w_aligned[i] and 
                close[i] <= ema_20[i] * 1.01 and  # within 1% above EMA20
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA21 and pulling back to daily EMA20 with volume
            elif (close[i] < ema_21_1w_aligned[i] and 
                  close[i] >= ema_20[i] * 0.99 and  # within 1% below EMA20
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly EMA21
            if close[i] < ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly EMA21
            if close[i] > ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals