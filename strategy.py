#!/usr/bin/env python3
name = "12h_Weekly_Camarilla_R3S3_Breakout_VolumeSpike"
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
    
    # Weekly high/low for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels (R3, S3)
    # R3 = Close + 1.1*(High - Low)/2
    # S3 = Close - 1.1*(High - Low)/2
    camarilla_width = 1.1 * (high_1w - low_1w) / 2
    r3_weekly = close_1w + camarilla_width
    s3_weekly = close_1w - camarilla_width
    
    # Align weekly Camarilla levels to 12h timeframe
    r3_weekly_aligned = align_htf_to_ltf(prices, df_1w, r3_weekly)
    s3_weekly_aligned = align_htf_to_ltf(prices, df_1w, s3_weekly)
    
    # Daily volume filter: volume > 1.8x 20-day average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    volume_filter = volume > 1.8 * vol_ma20_1d_aligned
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(r3_weekly_aligned[i]) or np.isnan(s3_weekly_aligned[i]) or
            np.isnan(vol_ma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above weekly R3 + volume filter
            if close[i] > r3_weekly_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below weekly S3 + volume filter
            elif close[i] < s3_weekly_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close below weekly S3
            if close[i] < s3_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close above weekly R3
            if close[i] > r3_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals