#!/usr/bin/env python3
name = "1d_1w_Camarilla_R3S3_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA34 for trend
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_1w = close_1w > ema34_1w
    
    # Get 1d data for Camarilla levels (R3, S3)
    high_1d = df_1w['high'].values  # Note: This is actually 1w data, but we need 1d data for Camarilla calculation
    low_1d = df_1w['low'].values
    close_1d = df_1w['close'].values
    # Correction: Need to get proper 1d data for Camarilla calculation
    # We'll get 1d data separately for price levels
    
    # Actually get 1d data for price levels and Camarilla calculation
    df_1d_price = get_htf_data(prices, '1d')
    if len(df_1d_price) < 20:
        return np.zeros(n)
    
    high_1d = df_1d_price['high'].values
    low_1d = df_1d_price['low'].values
    close_1d = df_1d_price['close'].values
    
    # Calculate 1w trend from 1w data
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_1w = close_1w > ema34_1w
    
    # Calculate Camarilla levels (R3, S3) from previous 1d period
    R3 = np.full(len(high_1d), np.nan)
    S3 = np.full(len(high_1d), np.nan)
    
    for i in range(1, len(high_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        if range_val > 0:
            R3[i] = prev_close + range_val * 1.1 / 4
            S3[i] = prev_close - range_val * 1.1 / 4
    
    # Volume moving average (20-period) for confirmation
    vol_ma20 = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            if i > 0:
                vol_ma20[i] = np.mean(volume[:i+1])
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    # Align indicators to 1d timeframe
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    R3_aligned = align_htf_to_ltf(prices, df_1d_price, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d_price, S3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 34)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or
            np.isnan(trend_up_1w_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Fixed position size: 0.25
        position_size = 0.25
        
        if position == 0:
            # Long: price breaks above R3 + uptrend + volume confirmation
            if (close[i] > R3_aligned[i] and 
                trend_up_1w_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = position_size
                position = 1
            # Short: price breaks below S3 + downtrend + volume confirmation
            elif (close[i] < S3_aligned[i] and 
                  not trend_up_1w_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -position_size
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 or trend changes
            if (close[i] < S3_aligned[i] or 
                not trend_up_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position_size
        elif position == -1:
            # Short exit: price breaks above R3 or trend changes
            if (close[i] > R3_aligned[i] or 
                trend_up_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -position_size
    
    return signals