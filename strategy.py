#!/usr/bin/env python3
name = "1h_4h_1d_Camarilla_Breakout_Trend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data for trend and pivot levels
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h trend: EMA50 > EMA200 for uptrend
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    trend_up_4h = ema50_4h > ema200_4h
    trend_down_4h = ema50_4h < ema200_4h
    
    # Calculate 1d Camarilla pivot points (using previous day)
    high_1d = df_1d['high'].shift(1).values
    low_1d = df_1d['low'].shift(1).values
    close_1d = df_1d['close'].shift(1).values
    range_1d = high_1d - low_1d
    pivot = (high_1d + low_1d + close_1d) / 3
    r3 = close_1d + range_1d * 1.1 / 2
    s3 = close_1d - range_1d * 1.1 / 2
    
    # Align 4h trend and 1d Camarilla levels to 1h
    trend_up_aligned = align_htf_to_ltf(prices, df_4h, trend_up_4h)
    trend_down_aligned = align_htf_to_ltf(prices, df_4h, trend_down_4h)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: current volume > 2.0 x 20-period average (more selective)
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 50)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN or outside session
        if (np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 in 4h uptrend with volume spike
            if (close[i] > r3_aligned[i] and 
                trend_up_aligned[i] and 
                volume[i] > 2.0 * vol_ma20[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 in 4h downtrend with volume spike
            elif (close[i] < s3_aligned[i] and 
                  trend_down_aligned[i] and 
                  volume[i] > 2.0 * vol_ma20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price falls below pivot or 4h trend changes
            if (close[i] < pivot_aligned[i] or not trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price rises above pivot or 4h trend changes
            if (close[i] > pivot_aligned[i] or not trend_down_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals