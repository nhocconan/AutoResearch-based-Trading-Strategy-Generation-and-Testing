#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_R3S3_Breakout_Trend_Volume_v2"
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
    
    # Get weekly data for pivot calculation
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 10:
        return np.zeros(n)
    
    # Previous week's OHLC for weekly pivot
    prev_high_w = df_w['high'].shift(1).values
    prev_low_w = df_w['low'].shift(1).values
    prev_close_w = df_w['close'].shift(1).values
    
    # Calculate weekly pivot points
    pivot_w = (prev_high_w + prev_low_w + prev_close_w) / 3
    r3_w = pivot_w + 1.1 * (prev_high_w - prev_low_w)
    s3_w = pivot_w - 1.1 * (prev_high_w - prev_low_w)
    
    # Get daily data for trend and volume filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 20:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema50_d = pd.Series(df_d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily volume filter: current volume > 1.5 * 20-day average
    vol_series_d = pd.Series(df_d['volume'].values)
    vol_ma_d = vol_series_d.rolling(window=20, min_periods=20).mean().values
    volume_filter_d = df_d['volume'].values > (vol_ma_d * 1.5)
    
    # Align all to 6h timeframe
    r3_w_6h = align_htf_to_ltf(prices, df_w, r3_w)
    s3_w_6h = align_htf_to_ltf(prices, df_w, s3_w)
    ema50_d_6h = align_htf_to_ltf(prices, df_d, ema50_d)
    volume_filter_6h = align_htf_to_ltf(prices, df_d, volume_filter_d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Need enough data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r3_w_6h[i]) or np.isnan(s3_w_6h[i]) or
            np.isnan(ema50_d_6h[i]) or np.isnan(volume_filter_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r3_val = r3_w_6h[i]
        s3_val = s3_w_6h[i]
        trend = ema50_d_6h[i]
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
            # Exit long: close below S3 (mean reversion to pivot)
            if close[i] < s3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above R3 (mean reversion to pivot)
            if close[i] > r3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals