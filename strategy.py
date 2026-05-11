#!/usr/bin/env python3
name = "1h_Camarilla_R3_S3_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivots and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels from previous 4h bar's OHLC
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Previous 4h bar's range
    range_4h = high_4h - low_4h
    
    # Calculate Camarilla R3 and S3 levels (most commonly used for breakouts)
    camarilla_r3 = close_4h + (range_4h * 1.1 / 2)
    camarilla_s3 = close_4h - (range_4h * 1.1 / 2)
    
    # Align Camarilla levels to 1h timeframe (using previous 4h bar's values)
    r3_1h = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    s3_1h = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # 4h EMA50 for trend filter
    close_4h_series = pd.Series(close_4h)
    ema_4h = close_4h_series.ewm(span=50, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume filter: current volume > 1.8x 24-period average (higher threshold = fewer trades)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_1h[i]) or np.isnan(s3_1h[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND above 4h EMA50 (uptrend) AND volume spike
            if close[i] > r3_1h[i] and close[i] > ema_4h_aligned[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 AND below 4h EMA50 (downtrend) AND volume spike
            elif close[i] < s3_1h[i] and close[i] < ema_4h_aligned[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price falls below S3 OR below 4h EMA50 (trend change)
            if close[i] < s3_1h[i] or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20  # maintain position
        elif position == -1:
            # Short exit: price rises above R3 OR above 4h EMA50 (trend change)
            if close[i] > r3_1h[i] or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20  # maintain position
    
    return signals