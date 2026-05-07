#!/usr/bin/env python3
# 12h_Camarilla_R3S3_Breakout_1dTrend_Volume
# Hypothesis: 12h strategy using daily Camarilla levels (R3/S3) with 1d EMA50 trend filter and volume spike.
# Enters long when price breaks above R3, price > EMA50 (uptrend), and volume > 2x average.
# Enters short when price breaks below S3, price < EMA50 (downtrend), and volume > 2x average.
# Exits when price returns to opposite level (S3 for long, R3 for short).
# Designed for 12h timeframe to limit trades (target: 50-150 total over 4 years) while capturing trends in both bull and bear markets.

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d period's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    hl_range = high_1d - low_1d
    r3_1d = close_1d + 1.1 * hl_range / 2
    s3_1d = close_1d - 1.1 * hl_range / 2
    
    # Calculate EMA50 for trend filter (1d)
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all to 12h timeframe (use previous 1d period's levels)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike detection: 2.0x average volume (30-period for stability)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 50)  # Ensure we have volume MA and EMA50 data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3, price above EMA50 (uptrend), volume spike (>2x)
            if (close[i] > r3_1d_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, price below EMA50 (downtrend), volume spike (>2x)
            elif (close[i] < s3_1d_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to or below S3 (opposite level)
            if close[i] <= s3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to or above R3 (opposite level)
            if close[i] >= r3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals