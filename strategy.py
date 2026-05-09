#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_R3S3_Breakout_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels and trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Weekly high, low, close for pivot calculation
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly pivot levels (R3, S3)
    range_weekly = high_weekly - low_weekly
    pivot_point = (high_weekly + low_weekly + close_weekly) / 3.0
    r3_level = pivot_point + 1.1 * range_weekly  # R3 = pivot + 1.1 * range
    s3_level = pivot_point - 1.1 * range_weekly  # S3 = pivot - 1.1 * range
    
    # Weekly EMA50 for trend filter
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly data to daily
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3_level)
    ema50_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Daily volume filter (20-period average)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        trend = ema50_aligned[i]
        vol_ok = volume[i] > vol_avg[i] * 1.5
        
        if position == 0:
            # Long: break above R3 with volume and above weekly EMA50
            if close[i] > r3 and vol_ok and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with volume and below weekly EMA50
            elif close[i] < s3 and vol_ok and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below S3 or trend reversal
            if close[i] < s3 or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above R3 or trend reversal
            if close[i] > r3 or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals