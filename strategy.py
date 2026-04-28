#!/usr/bin/env python3
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
    
    # Get daily data for HTF context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for additional context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily high/low for previous day (used for Camarilla)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly high/low for trend context
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate daily Camarilla pivot levels (based on previous day)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    r1 = pivot + range_1d * 1.1 / 12
    s1 = pivot - range_1d * 1.1 / 12
    r2 = pivot + range_1d * 1.1 / 6
    s2 = pivot - range_1d * 1.1 / 6
    r3 = pivot + range_1d * 1.1 / 4
    s3 = pivot - range_1d * 1.1 / 4
    
    # Calculate weekly trend using EMA
    weekly_close = df_1w['close'].values
    ema_20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_uptrend = weekly_close > ema_20
    weekly_downtrend = weekly_close < ema_20
    
    # Align daily Camarilla levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Align weekly trend to 12h timeframe
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1d, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1d, weekly_downtrend.astype(float))
    
    # Volume filter: above average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(weekly_uptrend_aligned[i]) or
            np.isnan(weekly_downtrend_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: above average volume
        vol_filter = volume[i] > vol_ma[i]
        
        # Entry conditions: Camarilla breakout with weekly trend filter
        # Long: break above R3 with weekly uptrend
        long_entry = (close[i] > r3_aligned[i]) and weekly_uptrend_aligned[i] > 0.5 and vol_filter
        # Short: break below S3 with weekly downtrend
        short_entry = (close[i] < s3_aligned[i]) and weekly_downtrend_aligned[i] > 0.5 and vol_filter
        
        # Exit conditions: opposite Camarilla level or loss of weekly trend
        long_exit = (close[i] < s3_aligned[i]) or weekly_uptrend_aligned[i] <= 0.5
        short_exit = (close[i] > r3_aligned[i]) or weekly_downtrend_aligned[i] <= 0.5
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R3S3_WeeklyTrend_Filter"
timeframe = "12h"
leverage = 1.0