#!/usr/bin/env python3
# Hypothesis: 12h timeframe with weekly trend filter (EMA50), daily Camarilla R3/S3 breakout,
# and volume confirmation. Targets 20-50 trades over 4 years (5-12/year) to minimize fee drag.
# Weekly EMA50 filters trend direction, daily Camarilla R3/S3 provides breakout levels,
# volume surge confirms momentum. Designed to work in both bull (breakouts) and bear (mean reversion via tight stops).
name = "12h_Camarilla_R3S3_WeeklyTrend_Volume"
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
    
    # Get weekly data for trend filter (EMA50)
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_w = df_w['close'].values
    ema_w = pd.Series(close_w).ewm(span=50, min_periods=50).mean().values
    ema_w_aligned = align_htf_to_ltf(prices, df_w, ema_w)
    
    # Get daily data for Camarilla R3/S3 levels (from previous day)
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 50:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # Previous day's range
    range_d = high_d - low_d
    
    # Calculate Camarilla R3 and S3 levels
    camarilla_r3 = close_d + (range_d * 1.1 / 4)
    camarilla_s3 = close_d - (range_d * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe (using previous day's values)
    r3_12h = align_htf_to_ltf(prices, df_d, camarilla_r3)
    s3_12h = align_htf_to_ltf(prices, df_d, camarilla_s3)
    
    # Volume filter: current volume > 2.0x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(ema_w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND above weekly EMA50 (uptrend) AND volume surge
            if close[i] > r3_12h[i] and close[i] > ema_w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND below weekly EMA50 (downtrend) AND volume surge
            elif close[i] < s3_12h[i] and close[i] < ema_w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below S3 OR below weekly EMA50 (trend change)
            if close[i] < s3_12h[i] or close[i] < ema_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above R3 OR above weekly EMA50 (trend change)
            if close[i] > r3_12h[i] or close[i] > ema_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals