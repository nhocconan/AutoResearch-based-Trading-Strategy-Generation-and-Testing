#!/usr/bin/env python3
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
    
    # Get daily data for pivot calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate daily range for pivot calculations (previous day's data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    daily_range = high_1d - low_1d
    
    # Weekly Camarilla pivot levels (based on previous week's data)
    # Using weekly high/low/close from previous week for current week's levels
    weekly_high = df_1d['high'].values  # Daily high for weekly aggregation
    weekly_low = df_1d['low'].values    # Daily low for weekly aggregation
    weekly_close = df_1d['close'].values # Daily close for weekly aggregation
    
    # Calculate weekly range (max high - min low over past 5 trading days)
    # For simplicity, using daily range as proxy - in practice would use weekly aggregation
    weekly_range = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values - \
                   pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    
    # Weekly Camarilla levels (R3/S3 levels)
    camarilla_r3 = weekly_close + weekly_range * 1.1 / 4
    camarilla_s3 = weekly_close - weekly_range * 1.1 / 4
    
    # Align Weekly Camarilla levels to daily timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Weekly trend filter: price above/below weekly SMA10
    close_1w_series = pd.Series(df_1w['close'].values)
    sma10_1w = close_1w_series.rolling(window=10, min_periods=10).mean().values
    sma10_1w_aligned = align_htf_to_ltf(prices, df_1w, sma10_1w)
    
    # Volume filter: above average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(sma10_1w_aligned[i]) or np.isnan(vol_ma[i])):
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
        
        # Trend filter: price above/below weekly SMA10
        trend_up = close[i] > sma10_1w_aligned[i]
        trend_down = close[i] < sma10_1w_aligned[i]
        
        # Entry conditions: 
        # Long: price breaks above weekly R3 with volume and trend up
        # Short: price breaks below weekly S3 with volume and trend down
        long_entry = (close[i] > r3_aligned[i]) and vol_filter and trend_up
        short_entry = (close[i] < s3_aligned[i]) and vol_filter and trend_down
        
        # Exit conditions: price returns to opposite weekly S3/R3 levels
        long_exit = (close[i] < s3_aligned[i])
        short_exit = (close[i] > r3_aligned[i])
        
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

name = "1d_WeeklyCamarilla_R3S3_WeeklyTrend_Volume_Session"
timeframe = "1d"
leverage = 1.0