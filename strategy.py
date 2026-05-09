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
    volume = prices['volume'].values
    
    # Get weekly data for trend and pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA50 for trend
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Weekly pivot levels (R3, S3) - using previous week's high/low/close
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    range_1w = high_1w - low_1w
    pivot = (high_1w + low_1w + close_1w) / 3
    r3 = pivot + 1.1 * (high_1w - low_1w)  # R3 = pivot + 1.1*(H-L)
    s3 = pivot - 1.1 * (high_1w - low_1w)  # S3 = pivot - 1.1*(H-L)
    
    # Align all to daily
    ema50_1w_d = align_htf_to_ltf(prices, df_1w, ema50_1w)
    r3_d = align_htf_to_ltf(prices, df_1w, r3)
    s3_d = align_htf_to_ltf(prices, df_1w, s3)
    
    # Volume filter: today's volume > 20-day average volume
    vol_avg_20d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1w_d[i]) or np.isnan(r3_d[i]) or np.isnan(s3_d[i]) or 
            np.isnan(vol_avg_20d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema50_1w_d[i]
        resistance = r3_d[i]
        support = s3_d[i]
        vol_ok = volume[i] > vol_avg_20d[i] * 1.5
        
        if position == 0:
            # Long: break above R3 with volume and above weekly EMA50
            if close[i] > resistance and vol_ok and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with volume and below weekly EMA50
            elif close[i] < support and vol_ok and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below S3 or trend reversal
            if close[i] < support or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above R3 or trend reversal
            if close[i] > resistance or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals