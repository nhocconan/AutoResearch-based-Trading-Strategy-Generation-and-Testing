#!/usr/bin/env python3
# 12h_Pivot_Reversal_1DTrend_Volume
# Hypothesis: Uses daily pivot points (standard) with price reversal from support/resistance, confirmed by 1-day EMA trend and volume spike. Designed for 12h timeframe to capture swing reversals in both bull and bear markets with low trade frequency.
# Target: 15-30 trades per year per symbol with clear entry/exit rules.

name = "12h_Pivot_Reversal_1DTrend_Volume"
timeframe = "12h"
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
    
    # Get 1d data for pivot calculation and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily OHLC for standard pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Standard pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # 1-day EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align pivot levels and EMA to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.8x average volume (30-period)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure we have volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema34_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation
        volume_filter = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: Price bounces off S1 (support) with Uptrend (price > EMA34) + volume spike
            if (low[i] <= s1_aligned[i] and close[i] > s1_aligned[i] and
                close[i] > ema34_aligned[i] and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price rejects R1 (resistance) with Downtrend (price < EMA34) + volume spike
            elif (high[i] >= r1_aligned[i] and close[i] < r1_aligned[i] and
                  close[i] < ema34_aligned[i] and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit conditions:
            # 1. Price crosses pivot point (mean reversion)
            # 2. Opposite rejection (long exits at R1, short exits at S1)
            pivot_cross = (position == 1 and close[i] < pivot_aligned[i]) or \
                          (position == -1 and close[i] > pivot_aligned[i])
            opposite_reject = (position == 1 and high[i] >= r1_aligned[i] and close[i] < r1_aligned[i]) or \
                              (position == -1 and low[i] <= s1_aligned[i] and close[i] > s1_aligned[i])
            
            if pivot_cross or opposite_reject:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals