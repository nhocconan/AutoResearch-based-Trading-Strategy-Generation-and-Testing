#!/usr/bin/env python3
name = "6h_1d_1w_WeeklyPivot_R1S1_Breakout_Trend_Volume"
timeframe = "6h"
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
    
    # Get 1d data for daily trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_1d = close_1d > ema34_1d
    
    # Get 1w data for weekly pivot levels (R1, S1)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (R1, S1) from previous week
    PP = np.zeros(len(high_1w))
    R1 = np.zeros(len(high_1w))
    S1 = np.zeros(len(high_1w))
    
    for i in range(len(high_1w)):
        if i < 1:
            PP[i] = np.nan
            R1[i] = np.nan
            S1[i] = np.nan
        else:
            prev_high = high_1w[i-1]
            prev_low = low_1w[i-1]
            prev_close = close_1w[i-1]
            PP[i] = (prev_high + prev_low + prev_close) / 3
            R1[i] = 2 * PP[i] - prev_low
            S1[i] = 2 * PP[i] - prev_high
    
    # Align indicators to 6h timeframe
    R1_1w_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_1w_aligned = align_htf_to_ltf(prices, df_1w, S1)
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    
    # Volume moving average (20-period) for confirmation
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(R1_1w_aligned[i]) or 
            np.isnan(S1_1w_aligned[i]) or
            np.isnan(trend_up_1d_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + uptrend + volume confirmation
            if (close[i] > R1_1w_aligned[i] and 
                trend_up_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + downtrend + volume confirmation
            elif (close[i] < S1_1w_aligned[i] and 
                  not trend_up_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 or trend changes
            if (close[i] < S1_1w_aligned[i] or not trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 or trend changes
            if (close[i] > R1_1w_aligned[i] or trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals