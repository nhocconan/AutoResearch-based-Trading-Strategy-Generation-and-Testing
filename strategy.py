#!/usr/bin/env python3
# 1D_Camarilla_R3S3_1WTrend_Volume_Signal
# Hypothesis: Daily timeframe strategy using weekly timeframe for trend and levels to reduce noise. Uses Camarilla R3/S3 breakouts with weekly trend filter and volume confirmation to capture multi-day moves in both bull and bear markets. Target: 15-25 trades/year per symbol to stay under 100 total trades over 4 years.

name = "1D_Camarilla_R3S3_1WTrend_Volume_Signal"
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
    
    # Get weekly data for trend and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly OHLC for Camarilla levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla R3 and S3 levels from weekly range
    hl_range_1w = high_1w - low_1w
    r3_1w = close_1w + 1.1 * hl_range_1w / 2
    s3_1w = close_1w - 1.1 * hl_range_1w / 2
    
    # Align weekly levels to daily timeframe (use previous weekly period's levels)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Weekly trend filter: EMA34 on weekly close
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation: 2.0x average volume (50-period for stability on daily)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # Ensure we have volume MA and weekly EMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R3, price above weekly EMA34 (uptrend), volume spike (>2.0x)
            if (close[i] > r3_1w_aligned[i] and 
                close[i] > ema34_1w_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S3, price below weekly EMA34 (downtrend), volume spike (>2.0x)
            elif (close[i] < s3_1w_aligned[i] and 
                  close[i] < ema34_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to or below weekly S3 (opposite level)
            if close[i] <= s3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to or above weekly R3 (opposite level)
            if close[i] >= r3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals