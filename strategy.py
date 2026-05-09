#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_R3S3_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Previous week's close, high, low for weekly pivot calculation (R3, S3)
    prev_week_close = df_1w['close'].shift(1).values
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    
    # Calculate weekly Camarilla levels (R3, S3)
    r3 = prev_week_close + 1.1 * (prev_week_high - prev_week_low) * 3 / 4
    s3 = prev_week_close - 1.1 * (prev_week_high - prev_week_low) * 3 / 4
    
    # Trend filter: weekly EMA34
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume filter: current 1d volume > 1.8 * 20-period average
    vol_series = pd.Series(df_1d['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = df_1d['volume'].values > (vol_ma * 1.8)
    
    # Align all to daily
    r3_1d = align_htf_to_ltf(prices, df_1w, r3)
    s3_1d = align_htf_to_ltf(prices, df_1w, s3)
    ema34_1w_1d = align_htf_to_ltf(prices, df_1w, ema34_1w)
    volume_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # Need enough data for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r3_1d[i]) or np.isnan(s3_1d[i]) or
            np.isnan(ema34_1w_1d[i]) or np.isnan(volume_filter_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r3_val = r3_1d[i]
        s3_val = s3_1d[i]
        trend = ema34_1w_1d[i]
        vol_filter = volume_filter_1d_aligned[i]
        
        if position == 0:
            # Enter long: break above R3 with volume and above weekly trend
            if close[i] > r3_val and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below S3 with volume and below weekly trend
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