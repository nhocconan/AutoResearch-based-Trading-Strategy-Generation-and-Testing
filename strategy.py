#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d weekly pivot levels with directional bias from 1w trend
# - Long when price breaks above weekly R3 with volume expansion and 1w trend up (price > 1w EMA50)
# - Short when price breaks below weekly S3 with volume expansion and 1w trend down (price < 1w EMA50)
# - Exit when price crosses back below/above weekly pivot (P)
# - Uses weekly pivot levels (calculated from prior week) to avoid look-ahead
# - Volume filter requires current volume > 1.5x 20-period average
# - Designed to capture strong breakouts in trending markets while avoiding false signals
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_WeeklyPivot_R3S3_Breakout_1wTrend_Volume"
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
    
    # Get 1d data for weekly pivot calculation (using prior week's data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week's OHLC
    # Weekly high/low/close: use rolling window of 5 days (1 week)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly high: max of high over past 5 days
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    # Weekly low: min of low over past 5 days
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    # Weekly close: close of 5th day ago (shifted by 5)
    weekly_close = np.roll(close_1d, 5)
    weekly_close[:5] = np.nan  # First 5 values invalid
    
    # Weekly pivot point: (H + L + C) / 3
    weekly_p = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly R3: P + 2*(H - L)
    weekly_r3 = weekly_p + 2.0 * (weekly_high - weekly_low)
    # Weekly S3: P - 2*(H - L)
    weekly_s3 = weekly_p - 2.0 * (weekly_high - weekly_low)
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 1d/1w indicators to 6h timeframe
    weekly_p_6h = align_htf_to_ltf(prices, df_1d, weekly_p)
    weekly_r3_6h = align_htf_to_ltf(prices, df_1d, weekly_r3)
    weekly_s3_6h = align_htf_to_ltf(prices, df_1d, weekly_s3)
    ema_50_1w_6h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filters (6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)  # Volume expansion
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(weekly_p_6h[i]) or np.isnan(weekly_r3_6h[i]) or np.isnan(weekly_s3_6h[i]) or 
            np.isnan(ema_50_1w_6h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above weekly R3 with volume expansion and 1w trend up
            if close[i] > weekly_r3_6h[i] and volume_filter[i] and close[i] > ema_50_1w_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below weekly S3 with volume expansion and 1w trend down
            elif close[i] < weekly_s3_6h[i] and volume_filter[i] and close[i] < ema_50_1w_6h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly pivot
            if close[i] < weekly_p_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above weekly pivot
            if close[i] > weekly_p_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals