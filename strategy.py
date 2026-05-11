#!/usr/bin/env python3
name = "1D_1W_Camarilla_Pivot_Breakout_Volume"
timeframe = "1d"
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
    
    # Get daily and weekly data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend
    weekly_close = df_1w['close'].values
    ema50_w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_trend = weekly_close > ema50_w
    
    # Daily Camarilla pivot levels (based on previous day)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close_prev = df_1d['close'].values
    
    pivot = (daily_high + daily_low + daily_close_prev) / 3
    range_hl = daily_high - daily_low
    
    # Camarilla levels
    r3 = pivot + (range_hl * 1.1 / 2)
    r4 = pivot + (range_hl * 1.1)
    s3 = pivot - (range_hl * 1.1 / 2)
    s4 = pivot - (range_hl * 1.1)
    
    # 20-period volume average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    # Align to daily timeframe
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    vol_ma20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(weekly_trend_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: weekly uptrend + price breaks above R3 + volume confirmation
            if (weekly_trend_aligned[i] and 
                close[i] > r3_aligned[i] and 
                volume[i] > 1.5 * vol_ma20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + price breaks below S3 + volume confirmation
            elif (not weekly_trend_aligned[i] and 
                  close[i] < s3_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly trend changes or price breaks below S3
            if (not weekly_trend_aligned[i] or close[i] < s3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: weekly trend changes or price breaks above R3
            if (weekly_trend_aligned[i] or close[i] > r3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals