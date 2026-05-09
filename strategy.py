#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_R3S3_Breakout_1wTrend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Previous day's data for daily pivot (R3, S3)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate daily pivot and R3/S3 levels
    pivot = (prev_high + prev_low + prev_close) / 3
    r3 = pivot + 2 * (prev_high - prev_low)  # R3 = Pivot + 2*(H-L)
    s3 = pivot - 2 * (prev_high - prev_low)  # S3 = Pivot - 2*(H-L)
    
    # Weekly EMA50 trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to daily timeframe
    r3_1d = align_htf_to_ltf(prices, df_1d, r3)
    s3_1d = align_htf_to_ltf(prices, df_1d, s3)
    ema50_1w_1d = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 30)  # Need enough data for weekly EMA50
    
    for i in range(start_idx, n):
        if (np.isnan(r3_1d[i]) or np.isnan(s3_1d[i]) or 
            np.isnan(ema50_1w_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r3_val = r3_1d[i]
        s3_val = s3_1d[i]
        trend = ema50_1w_1d[i]
        
        if position == 0:
            # Enter long: break above R3 with weekly uptrend
            if close[i] > r3_val and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Enter short: break below S3 with weekly downtrend
            elif close[i] < s3_val and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below pivot (mean reversion)
            if close[i] < pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above pivot (mean reversion)
            if close[i] > pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals