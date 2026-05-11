#!/usr/bin/env python3
name = "1h_Camarilla_4hTrend_1dVol"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (EMA)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range
    range_1d = high_1d - low_1d
    
    # Camarilla R3 and S3 (most relevant for breakouts)
    camarilla_r3 = close_1d + (range_1d * 1.1 / 2)
    camarilla_s3 = close_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 1h timeframe
    r3_1h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_1h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 4h EMA for trend filter (21-period for responsiveness)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, min_periods=21).mean().values
    ema_4h_1h = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d volume filter: current day volume > 1.5x 20-day average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = vol_1d / vol_ma_1d
    vol_ratio_1h = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_1h[i]) or np.isnan(s3_1h[i]) or 
            np.isnan(ema_4h_1h[i]) or np.isnan(vol_ratio_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Only trade during session
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND above 4h EMA (uptrend) AND volume spike
            if close[i] > r3_1h[i] and close[i] > ema_4h_1h[i] and vol_ratio_1h[i] > 1.5:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 AND below 4h EMA (downtrend) AND volume spike
            elif close[i] < s3_1h[i] and close[i] < ema_4h_1h[i] and vol_ratio_1h[i] > 1.5:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price falls below S3 OR below 4h EMA (trend change)
            if close[i] < s3_1h[i] or close[i] < ema_4h_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20  # maintain position
        elif position == -1:
            # Short exit: price rises above R3 OR above 4h EMA (trend change)
            if close[i] > r3_1h[i] or close[i] > ema_4h_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20  # maintain position
    
    return signals