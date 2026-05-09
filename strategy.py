#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_R3S3_Breakout_1dTrend_VolumeSurge"
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
    
    # Get weekly data for pivot points (weekly high/low/close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for trend and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Previous week's high, low, close for weekly pivot
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    # Calculate weekly pivot points
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    r3 = pivot + 1.1 * (prev_week_high - prev_week_low)
    s3 = pivot - 1.1 * (prev_week_high - prev_week_low)
    
    # Trend filter: daily EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume surge filter: current daily volume > 2.0 * 20-day average
    vol_series = pd.Series(df_1d['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_surge = df_1d['volume'].values > (vol_ma * 2.0)
    
    # Align all to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1w, r3)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3)
    ema34_1d_6h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    volume_surge_6h = align_htf_to_ltf(prices, df_1d, volume_surge)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # Need enough data for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or
            np.isnan(ema34_1d_6h[i]) or np.isnan(volume_surge_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r3_val = r3_6h[i]
        s3_val = s3_6h[i]
        trend = ema34_1d_6h[i]
        vol_surge = volume_surge_6h[i]
        
        if position == 0:
            # Enter long: break above R3 with volume surge and above daily trend
            if close[i] > r3_val and close[i] > trend and vol_surge:
                signals[i] = 0.25
                position = 1
            # Enter short: break below S3 with volume surge and below daily trend
            elif close[i] < s3_val and close[i] < trend and vol_surge:
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