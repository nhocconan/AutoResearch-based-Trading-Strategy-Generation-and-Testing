#!/usr/bin/env python3
# 12h_Camarilla_R3S3_Breakout_1wTrend_Volume
# Hypothesis: 12h strategy using weekly Camarilla levels with trend filter from 1w EMA50 and volume spike.
# Breaks out at R3/S3 levels when price > EMA50 (uptrend) or < EMA50 (downtrend) with volume > 2x average.
# Uses tight exits at opposite S3/R3 levels. Weekly trend filter reduces whipsaw in bear markets.

name = "12h_Camarilla_R3S3_Breakout_1wTrend_Volume"
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
    
    # Get weekly data for Camarilla pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous weekly OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate all Camarilla levels
    hl_range = high_1w - low_1w
    r3_1w = close_1w + 1.1 * hl_range / 2
    r2_1w = close_1w + 1.1 * hl_range / 6
    r1_1w = close_1w + 1.1 * hl_range / 12
    s1_1w = close_1w - 1.1 * hl_range / 12
    s2_1w = close_1w - 1.1 * hl_range / 6
    s3_1w = close_1w - 1.1 * hl_range / 2
    pp_1w = (high_1w + low_1w + close_1w) / 3
    
    # Align all levels to 12h timeframe (use previous week's levels)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    
    # Calculate EMA50 for trend filter (weekly)
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike detection: 2.0x average volume (30-period for stability)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 50)  # Ensure we have volume MA and EMA50 data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3, price above EMA50 (uptrend), volume spike (>2x)
            if (close[i] > r3_1w_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, price below EMA50 (downtrend), volume spike (>2x)
            elif (close[i] < s3_1w_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to or below S3 (opposite level)
            if close[i] <= s3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to or above R3 (opposite level)
            if close[i] >= r3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals