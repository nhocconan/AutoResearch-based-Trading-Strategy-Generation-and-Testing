#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Calculate previous 12h bar's Camarilla R3 and S3 levels
    prev_high_12h = df_12h['high'].shift(1).values
    prev_low_12h = df_12h['low'].shift(1).values
    prev_close_12h = df_12h['close'].shift(1).values
    
    # Camarilla levels: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    camarilla_r3 = prev_close_12h + (prev_high_12h - prev_low_12h) * 1.1 / 4
    camarilla_s3 = prev_close_12h - (prev_high_12h - prev_low_12h) * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # Get 1d data for trend filter (20-period EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA20 for trend filter
    ema20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Volume filter: current 12h volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 20)  # Need 20 periods for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema20_1d_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        trend = ema20_1d_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: break above R3 + above 1d EMA20 trend + volume filter
            if close[i] > r3 and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below S3 + below 1d EMA20 trend + volume filter
            elif close[i] < s3 and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below S3
            if close[i] < s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above R3
            if close[i] > r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals