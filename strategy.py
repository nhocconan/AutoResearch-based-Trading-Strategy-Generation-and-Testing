#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = pd.Series(df_4h['close'].values)
    ema50_4h = close_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume MA(20)
    vol_1d = pd.Series(df_1d['volume'].values)
    vol_ma20_1d = vol_1d.rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Calculate daily pivot points from previous day's OHLC
    # Use 1d data to get previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R3, S3)
    # R3 = Close + 1.1*(High - Low)
    # S3 = Close - 1.1*(High - Low)
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align to 1h timeframe (use previous day's levels to avoid look-ahead)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Use previous day's levels
    r3_prev = np.roll(camarilla_r3_aligned, 1)
    s3_prev = np.roll(camarilla_s3_aligned, 1)
    r3_prev[0] = np.nan
    s3_prev[0] = np.nan
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(r3_prev[i]) or 
            np.isnan(s3_prev[i]) or np.isnan(vol_ma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        vol_ok = volume[i] > 2.0 * vol_ma20_1d_aligned[i]
        
        if position == 0 and in_session:
            # Long: Close breaks above R3 with volume spike and above 4h EMA trend
            if close[i] > r3_prev[i] and vol_ok and close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: Close breaks below S3 with volume spike and below 4h EMA trend
            elif close[i] < s3_prev[i] and vol_ok and close[i] < ema50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses back below S3 (mean reversion)
            if close[i] < s3_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Price crosses back above R3
            if close[i] > r3_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals