#!/usr/bin/env python3
# 1D_Camarilla_R3_S3_WeeklyTrend_Volume
# Hypothesis: Weekly trend filter with daily Camarilla R3/S3 breakout and volume confirmation.
# Uses weekly EMA40 for trend direction to reduce false signals in sideways markets.
# Targets 10-20 trades/year to minimize fee drag while capturing major trends.
# Works in bull/bear via weekly trend filter and volatility-adjusted position sizing.

name = "1D_Camarilla_R3_S3_WeeklyTrend_Volume"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate weekly EMA40 for trend filter
    close_1w = df_1w['close'].values
    ema40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # Calculate daily Camarilla pivot levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels
    hl_range = high_1d - low_1d
    r3_1d = close_1d + 1.1 * hl_range / 2
    s3_1d = close_1d - 1.1 * hl_range / 2
    
    # Align Camarilla levels to daily timeframe (use previous day's levels)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Volume spike detection: 2.5x average volume (100-period for stability)
    vol_ma = pd.Series(volume).rolling(window=100, min_periods=100).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 40)  # Ensure we have volume MA and weekly EMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(ema40_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3, price above weekly EMA40 (uptrend), volume spike (>2.5x)
            if (close[i] > r3_1d_aligned[i] and 
                close[i] > ema40_1w_aligned[i] and 
                volume[i] > 2.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, price below weekly EMA40 (downtrend), volume spike (>2.5x)
            elif (close[i] < s3_1d_aligned[i] and 
                  close[i] < ema40_1w_aligned[i] and 
                  volume[i] > 2.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to or below S3 (mean reversion to center)
            if close[i] <= s3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to or above R3 (mean reversion to center)
            if close[i] >= r3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals