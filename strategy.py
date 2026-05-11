#!/usr/bin/env python3
name = "12h_1d_Camarilla_R3S3_Breakout_Volume_Trend"
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
    
    # Get daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily close for trend
    daily_close = df_1d['close'].values
    ema34_d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_trend = daily_close > ema34_d
    
    # Calculate Camarilla levels from previous day
    # Camarilla: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    # But we use R3/S3 levels: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # Actually standard Camarilla: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close_prev = df_1d['close'].values
    
    # Calculate R3 and S3 for each day
    camarilla_r3 = daily_close_prev + 1.1 * (daily_high - daily_low) / 2
    camarilla_s3 = daily_close_prev - 1.1 * (daily_high - daily_low) / 2
    
    # 20-period volume average for confirmation
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    # Align daily data to 12h
    daily_trend_aligned = align_htf_to_ltf(prices, df_1d, daily_trend)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(daily_trend_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: daily uptrend + price breaks above R3 + volume confirmation
            if (daily_trend_aligned[i] and 
                close[i] > camarilla_r3_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: daily downtrend + price breaks below S3 + volume confirmation
            elif (not daily_trend_aligned[i] and 
                  close[i] < camarilla_s3_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below S3 or trend changes
            if (close[i] < camarilla_s3_aligned[i] or not daily_trend_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above R3 or trend changes
            if (close[i] > camarilla_r3_aligned[i] or daily_trend_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals