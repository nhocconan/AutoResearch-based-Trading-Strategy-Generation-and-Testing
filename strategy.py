#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_R3S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Previous week's close, high, low for Camarilla (R3, S3)
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    
    # Calculate Weekly Camarilla levels (R3, S3)
    r3 = prev_close + 1.1 * (prev_high - prev_low) * 3 / 4
    s3 = prev_close - 1.1 * (prev_high - prev_low) * 3 / 4
    
    # Trend filter: weekly EMA50
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current daily volume > 1.5 * 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Align all to daily
    r3_1d = align_htf_to_ltf(prices, df_1w, r3)
    s3_1d = align_htf_to_ltf(prices, df_1w, s3)
    ema50_1w_1d = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Need enough data for EMA50
    
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
        vol_filter = volume_filter[i]
        
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