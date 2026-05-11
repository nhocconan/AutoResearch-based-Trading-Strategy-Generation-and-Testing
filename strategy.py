#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly and daily data
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 10 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Weekly Camarilla pivot levels (using previous week)
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    r3 = pivot + 1.1 * (prev_week_high - prev_week_low)
    s3 = pivot - 1.1 * (prev_week_high - prev_week_low)
    
    # Weekly trend filter: EMA14 > EMA34
    ema14_1w = pd.Series(df_1w['close']).ewm(span=14, adjust=False, min_periods=14).mean().values
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_1w = ema14_1w > ema34_1w
    trend_down_1w = ema14_1w < ema34_1w
    
    # Align all to 12h
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    trend_down_aligned = align_htf_to_ltf(prices, df_1w, trend_down_1w)
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 in weekly uptrend with volume surge
            if (close[i] > r3_aligned[i] and 
                trend_up_aligned[i] and 
                volume[i] > 2.0 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 in weekly downtrend with volume surge
            elif (close[i] < s3_aligned[i] and 
                  trend_down_aligned[i] and 
                  volume[i] > 2.0 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below pivot or weekly trend changes
            if (close[i] < pivot[i] if not np.isnan(pivot[i]) else False or not trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above pivot or weekly trend changes
            if (close[i] > pivot[i] if not np.isnan(pivot[i]) else False or not trend_down_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals