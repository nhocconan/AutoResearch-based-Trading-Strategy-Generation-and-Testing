#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_WeeklyPivot_R3S3_Breakout_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivots (using daily data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points using 1d data (weekly high/low/close)
    # We need to group by week to get weekly OHLC
    df_1d_copy = df_1d.copy()
    df_1d_copy['week'] = df_1d_copy.index.isocalendar().week
    df_1d_copy['year'] = df_1d_copy.index.isocalendar().year
    
    # Group by week to get weekly OHLC
    weekly = df_1d_copy.groupby(['year', 'week']).agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    if len(weekly) < 2:
        return np.zeros(n)
    
    # Calculate pivot points for each week (using prior week's data)
    weekly['pp'] = (weekly['high'].shift(1) + weekly['low'].shift(1) + weekly['close'].shift(1)) / 3
    weekly['r3'] = weekly['pp'] + 2 * (weekly['high'].shift(1) - weekly['low'].shift(1))
    weekly['s3'] = weekly['pp'] - 2 * (weekly['high'].shift(1) - weekly['low'].shift(1))
    
    # Map weekly data back to daily data
    df_1d['week'] = df_1d.index.isocalendar().week
    df_1d['year'] = df_1d.index.isocalendar().year
    df_1d = df_1d.merge(weekly[['year', 'week', 'r3', 's3']], on=['year', 'week'], how='left')
    
    # Forward fill to propagate weekly values to all days in the week
    df_1d['r3'] = df_1d['r3'].ffill()
    df_1d['s3'] = df_1d['s3'].ffill()
    
    r3 = df_1d['r3'].values
    s3 = df_1d['s3'].values
    
    # Trend filter: 1d EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current 1d volume > 1.3 * 20-day average
    vol_series = pd.Series(df_1d['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = df_1d['volume'].values > (vol_ma * 1.3)
    
    # Align all to 6h
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    ema50_1d_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_filter_6h = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Need enough data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or
            np.isnan(ema50_1d_6h[i]) or np.isnan(volume_filter_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r3_val = r3_6h[i]
        s3_val = s3_6h[i]
        trend = ema50_1d_6h[i]
        vol_filter = volume_filter_6h[i]
        
        if position == 0:
            # Enter long: break above R3 with volume and above trend
            if close[i] > r3_val and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below S3 with volume and below trend
            elif close[i] < s3_val and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below S3 (mean reversion to center)
            if close[i] < s3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above R3 (mean reversion to center)
            if close[i] > r3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals