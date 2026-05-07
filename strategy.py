#!/usr/bin/env python3
# 1H_Camarilla_R3S3_4HTrend_Volume_Entry
# Hypothesis: Combine 4h trend direction with 1h entry timing using Camarilla R3/S3 levels from 4h.
# Use 4h EMA34 for trend filter, and volume spike on 1h for entry confirmation.
# Target: 15-37 trades/year per symbol by requiring 4h trend alignment + volume spike.
# Works in bull (trend following) and bear (counter-trend at extremes) via trend filter.

name = "1H_Camarilla_R3S3_4HTrend_Volume_Entry"
timeframe = "1h"
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
    
    # Get 4h data for Camarilla pivot calculation and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla R3 and S3 levels from 4h OHLC
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    hl_range_4h = high_4h - low_4h
    r3_4h = close_4h + 1.1 * hl_range_4h / 2
    s3_4h = close_4h - 1.1 * hl_range_4h / 2
    
    # Calculate EMA34 for trend filter on 4h
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 4h levels and EMA to 1h timeframe
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Volume spike detection: 2.0x average volume (20-period for responsiveness)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Ensure we have volume MA and EMA34 data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or 
            np.isnan(ema34_4h_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h R3, price above 4h EMA34 (uptrend), volume spike (>2.0x)
            if (close[i] > r3_4h_aligned[i] and 
                close[i] > ema34_4h_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h S3, price below 4h EMA34 (downtrend), volume spike (>2.0x)
            elif (close[i] < s3_4h_aligned[i] and 
                  close[i] < ema34_4h_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price returns to or below 4h S3 (opposite level)
            if close[i] <= s3_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price returns to or above 4h R3 (opposite level)
            if close[i] >= r3_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals