#!/usr/bin/env python3
# 6h_LongTermTrend_VolumeBreakout
# Hypothesis: Use 6h price closing above 6-month high with volume confirmation to capture long-term trends.
# Works in both bull and bear markets by only taking long positions during confirmed uptrends.
# Uses 1d timeframe for trend confirmation and volume spike for entry confirmation.
# Target: 15-35 trades per year (60-140 over 4 years) with low frequency to minimize fee drag.

name = "6h_LongTermTrend_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for trend confirmation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 6-month (~180-day) high on daily timeframe
    period_6m = 180
    high_6m = pd.Series(high_1d).rolling(window=period_6m, min_periods=period_6m).max().values
    
    # Align 6m high to 6h timeframe
    high_6m_aligned = align_htf_to_ltf(prices, df_1d, high_6m)
    
    # Calculate 6-month low for stop/reference
    low_6m = pd.Series(low_1d).rolling(window=period_6m, min_periods=period_6m).min().values
    low_6m_aligned = align_htf_to_ltf(prices, df_1d, low_6m)
    
    # Volume confirmation: 20-period average on 6h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(high_6m_aligned[i]) or np.isnan(low_6m_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above 6-month high (new long-term high)
        new_high = close[i] > high_6m_aligned[i]
        
        # Volume filter: volume above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks 6-month high with volume confirmation
            if new_high and vol_ok:
                signals[i] = 0.25
                position = 1
        elif position == 1:
            # EXIT: Price drops back below 6-month low (trend failure)
            if close[i] < low_6m_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals