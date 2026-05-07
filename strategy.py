#!/usr/bin/env python3
# 6h_Camarilla_R3S3_Breakout_12hTrend_Volume
# Hypothesis: 6s strategy using 12-hour Camarilla levels with trend filter from 12h EMA50 and volume spike.
# Breaks out at R3/S3 levels when price > EMA50 (uptrend) or < EMA50 (downtrend) with volume > 2x average.
# Uses tight exits at opposite S3/R3 levels to limit holding periods. Designed for 6h timeframe to avoid
# overtrading while capturing trends in both bull and bear markets via trend filter.

name = "6h_Camarilla_R3S3_Breakout_12hTrend_Volume"
timeframe = "6h"
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
    
    # Get 12h data for Camarilla pivot calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 12h period's OHLC
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate all Camarilla levels
    hl_range = high_12h - low_12h
    r3_12h = close_12h + 1.1 * hl_range / 2
    r2_12h = close_12h + 1.1 * hl_range / 6
    r1_12h = close_12h + 1.1 * hl_range / 12
    s1_12h = close_12h - 1.1 * hl_range / 12
    s2_12h = close_12h - 1.1 * hl_range / 6
    s3_12h = close_12h - 1.1 * hl_range / 2
    pp_12h = (high_12h + low_12h + close_12h) / 3
    
    # Align all levels to 6h timeframe (use previous 12h period's levels)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    r2_12h_aligned = align_htf_to_ltf(prices, df_12h, r2_12h)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    s2_12h_aligned = align_htf_to_ltf(prices, df_12h, s2_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    pp_12h_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    
    # Calculate EMA50 for trend filter (12h)
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike detection: 2.0x average volume (30-period for stability)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 50)  # Ensure we have volume MA and EMA50 data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3, price above EMA50 (uptrend), volume spike (>2x)
            if (close[i] > r3_12h_aligned[i] and 
                close[i] > ema50_12h_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, price below EMA50 (downtrend), volume spike (>2x)
            elif (close[i] < s3_12h_aligned[i] and 
                  close[i] < ema50_12h_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to or below S3 (opposite level)
            if close[i] <= s3_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to or above R3 (opposite level)
            if close[i] >= r3_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals