#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_1w = close_1w > ema34_1w
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each daily bar
    R3 = np.zeros(len(df_1d))
    S3 = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        R3[i] = close_1d[i] + (high_1d[i] - low_1d[i]) * 1.1 / 6
        S3[i] = close_1d[i] - (high_1d[i] - low_1d[i]) * 1.1 / 6
    
    # Align Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(trend_up_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close breaks above R3 + weekly uptrend + volume confirmation
            if close[i] > R3_aligned[i] and trend_up_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S3 + weekly downtrend + volume confirmation
            elif close[i] < S3_aligned[i] and not trend_up_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close breaks below S3 OR weekly trend turns down
            if close[i] < S3_aligned[i] or not trend_up_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close breaks above R3 OR weekly trend turns up
            if close[i] > R3_aligned[i] or trend_up_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals