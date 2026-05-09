#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_S3_R3_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1h = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get daily data for Camarilla pivot levels (S3, R3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    daily_range = high_1d - low_1d
    pivot = (high_1d + low_1d + close_1d) / 3
    r3 = pivot + 1.1 * daily_range / 4
    s3 = pivot - 1.1 * daily_range / 4
    r3_1h = align_htf_to_ltf(prices, df_1d, r3)
    s3_1h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Daily volume average (20-period) for volume filter
    vol_avg_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_1h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1h[i]) or np.isnan(r3_1h[i]) or np.isnan(s3_1h[i]) or 
            np.isnan(vol_avg_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5 x daily 20-period average
        vol_spike = volume[i] > vol_avg_1h[i] * 1.5
        
        if position == 0:
            # Long: Break above Camarilla R3 with uptrend (price > 4h EMA50) and volume spike, in session
            if in_session[i] and close[i] > r3_1h[i] and close[i] > ema50_1h[i] and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: Break below Camarilla S3 with downtrend (price < 4h EMA50) and volume spike, in session
            elif in_session[i] and close[i] < s3_1h[i] and close[i] < ema50_1h[i] and vol_spike:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below Camarilla S3 OR trend turns down (price < 4h EMA50)
            if close[i] < s3_1h[i] or close[i] < ema50_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Price rises back above Camarilla R3 OR trend turns up (price > 4h EMA50)
            if close[i] > r3_1h[i] or close[i] > ema50_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals