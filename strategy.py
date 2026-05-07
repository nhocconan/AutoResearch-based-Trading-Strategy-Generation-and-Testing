#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dTrend_Volume_Tight_v2
Hypothesis: Tighten entry criteria from v1 by requiring volume > 2x average (vs 1.5x) and adding a 4-hour trend filter using EMA20 to avoid whipsaws. Target 15-25 trades/year by requiring stronger confluence. Uses 1d Camarilla R3/S3 for breakout levels, 1d EMA50 for trend filter, and volume spike for confirmation. Works in bull markets (breakouts with trend) and bear markets (fades from extremes with trend).
"""
name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume_Tight_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and Camarilla pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day OHLC for Camarilla pivot
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot levels calculation
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r3_1d = close_1d + (range_1d * 1.1 / 2)
    s3_1d = close_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # 1-day EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4-hour EMA20 for additional trend filter
    ema_20_4h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: current volume > 2.0 * 100-period average (stricter than v1)
    vol_avg = pd.Series(volume).rolling(window=100, min_periods=100).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup for EMA50 and volume average
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_20_4h[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + 1d uptrend + 4h uptrend + volume filter
            if (close[i] > r3_1d_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                close[i] > ema_20_4h[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + 1d downtrend + 4h downtrend + volume filter
            elif (close[i] < s3_1d_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  close[i] < ema_20_4h[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to opposite Camarilla level (S3 for long, R3 for short)
            if position == 1:
                if close[i] <= s3_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] >= r3_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals