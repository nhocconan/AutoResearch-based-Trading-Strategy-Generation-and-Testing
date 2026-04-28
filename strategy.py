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
    
    # Get daily data once for HTF context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for additional context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate daily high/low for pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly high/low for trend context
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Daily pivot points (based on previous day)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # R3 and S3 levels (key reversal levels)
    r3 = pivot + range_1d * 1.1 / 4
    s3 = pivot - range_1d * 1.1 / 4
    
    # Weekly trend: higher highs and higher lows = uptrend
    # Lower highs and lower lows = downtrend
    # Use 5-period smoothed weekly high/low
    high_1w_series = pd.Series(high_1w)
    low_1w_series = pd.Series(low_1w)
    high_ma = high_1w_series.rolling(window=5, min_periods=5).mean().values
    low_ma = low_1w_series.rolling(window=5, min_periods=5).mean().values
    
    # Align weekly moving averages to 4h timeframe
    high_ma_aligned = align_htf_to_ltf(prices, df_1w, high_ma)
    low_ma_aligned = align_htf_to_ltf(prices, df_1w, low_ma)
    
    # Weekly trend: price above both MA = uptrend, below both = downtrend
    weekly_uptrend = (high_ma_aligned > low_ma_aligned)  # Higher highs and higher lows
    weekly_downtrend = (high_ma_aligned < low_ma_aligned)  # Lower highs and lower lows
    
    # Align daily pivot levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: above average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_ma_aligned[i]) or np.isnan(low_ma_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i])):
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
        is_uptrend = weekly_uptrend[i]
        is_downtrend = weekly_downtrend[i]
        
        # Entry conditions: R3/S3 break with weekly trend alignment
        # Long: break above R3 in weekly uptrend
        long_entry = (close[i] > r3_aligned[i]) and is_uptrend and vol_filter
        # Short: break below S3 in weekly downtrend
        short_entry = (close[i] < s3_aligned[i]) and is_downtrend and vol_filter
        
        # Exit conditions: opposite level or loss of trend
        long_exit = (close[i] < s3_aligned[i]) or not is_uptrend
        short_exit = (close[i] > r3_aligned[i]) or not is_downtrend
        
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

name = "4h_Pivot_R3S3_WeeklyTrend_Filter"
timeframe = "4h"
leverage = 1.0