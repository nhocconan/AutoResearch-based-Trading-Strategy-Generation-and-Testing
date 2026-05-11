#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "12h"
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
    
    # Get weekly data for trend filter and 1d data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 10 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA21 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_1w = close_1w_series.ewm(span=21, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 1d Camarilla R3 and S3 levels from previous 1d bar's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    range_1d = high_1d - low_1d
    camarilla_r3 = close_1d + (range_1d * 1.1 / 2)
    camarilla_s3 = close_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume filter: current volume > 1.8x 30-period average (higher threshold = fewer trades)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND above weekly EMA21 (uptrend) AND volume spike
            if close[i] > r3_12h[i] and close[i] > ema_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND below weekly EMA21 (downtrend) AND volume spike
            elif close[i] < s3_12h[i] and close[i] < ema_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below S3 OR below weekly EMA21 (trend change)
            if close[i] < s3_12h[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above R3 OR above weekly EMA21 (trend change)
            if close[i] > r3_12h[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals