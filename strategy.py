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
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for trend context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate daily high/low for Camarilla pivot points (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily Camarilla pivot levels (based on previous day)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    r3 = pivot + range_1d * 1.1 / 4
    s3 = pivot - range_1d * 1.1 / 4
    r4 = pivot + range_1d * 1.1 / 2
    s4 = pivot - range_1d * 1.1 / 2
    
    # Align daily Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Weekly trend: price above/below weekly midpoint
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_range = high_1w - low_1w
    weekly_range_aligned = align_htf_to_ltf(prices, df_1w, weekly_range)
    low_1w_aligned = align_htf_to_ltf(prices, df_1w, low_1w)
    weekly_midpoint = low_1w_aligned + weekly_range_aligned / 2
    
    # Volume filter: above average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(weekly_range_aligned[i]) or np.isnan(weekly_midpoint[i]) or
            np.isnan(vol_ma[i])):
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
        
        # Weekly trend filter
        price_above_weekly_mid = close[i] > weekly_midpoint[i]
        price_below_weekly_mid = close[i] < weekly_midpoint[i]
        
        # Entry conditions: Camarilla breakout with weekly trend filter
        # Long: break above R3 with weekly uptrend
        long_entry = (close[i] > r3_aligned[i]) and price_above_weekly_mid and vol_filter
        # Short: break below S3 with weekly downtrend
        short_entry = (close[i] < s3_aligned[i]) and price_below_weekly_mid and vol_filter
        
        # Exit conditions: opposite Camarilla level or loss of weekly trend
        long_exit = (close[i] < s3_aligned[i]) or not price_above_weekly_mid
        short_exit = (close[i] > r3_aligned[i]) or not price_below_weekly_mid
        
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

name = "4h_Camarilla_R3S3_WeeklyTrend_Filter"
timeframe = "4h"
leverage = 1.0