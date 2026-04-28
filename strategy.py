#!/usr/bin/env python3
"""
12h_PivotReversal_1wTrend_VolumeFilter
Hypothesis: Daily pivot reversals with weekly trend filter and volume spike capture turning points in both bull and bear markets. Pivots provide objective support/resistance, weekly trend filters counter-trend noise, and volume confirms momentum. Targets 12-30 trades/year.
"""

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
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily pivot points: (H + L + C) / 3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align pivot and weekly EMA to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: >2.0x 30-period MA
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Pivot reversal conditions
        # Long: price crosses above pivot from below in downtrend
        # Short: price crosses below pivot from above in uptrend
        long_signal = (close[i] > pivot_aligned[i]) and (close[i-1] <= pivot_aligned[i-1]) and downtrend
        short_signal = (close[i] < pivot_aligned[i]) and (close[i-1] >= pivot_aligned[i-1]) and uptrend
        
        # Volume confirmation
        vol_confirm = volume[i] > (2.0 * vol_ma_30[i])
        
        # Entry logic: pivot reversal with volume and counter-trend
        long_entry = long_signal and vol_confirm
        short_entry = short_signal and vol_confirm
        
        # Exit logic: opposite pivot cross or trend alignment
        long_exit = (close[i] < pivot_aligned[i]) and (close[i-1] >= pivot_aligned[i-1])
        short_exit = (close[i] > pivot_aligned[i]) and (close[i-1] <= pivot_aligned[i-1])
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_PivotReversal_1wTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0