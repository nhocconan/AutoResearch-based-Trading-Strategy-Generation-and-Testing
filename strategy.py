#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume_v2"
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
    
    # Get 1d data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous day's close for Camarilla calculation (R3, S3)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels (R3, S3)
    r3 = prev_close + 1.1 * (prev_high - prev_low) * 3 / 4
    s3 = prev_close - 1.1 * (prev_high - prev_low) * 3 / 4
    
    # Trend filter: 1d EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current 12h volume > 1.5 * 20-period average
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    vol_series = pd.Series(df_12h['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter_12h = df_12h['volume'].values > (vol_ma * 1.5)
    
    # Align all to 12h
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    ema50_1d_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_filter_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_filter_12h)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Need enough data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or
            np.isnan(ema50_1d_12h[i]) or np.isnan(volume_filter_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r3_val = r3_12h[i]
        s3_val = s3_12h[i]
        trend = ema50_1d_12h[i]
        vol_filter = volume_filter_12h_aligned[i]
        
        if position == 0:
            # Enter long: break above R3 with volume and above trend
            if close[i] > r3_val and close[i] > trend and vol_filter:
                signals[i] = 0.30
                position = 1
            # Enter short: break below S3 with volume and below trend
            elif close[i] < s3_val and close[i] < trend and vol_filter:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit long: close below S3 (mean reversion to center)
            if close[i] < s3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: close above R3 (mean reversion to center)
            if close[i] > r3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals