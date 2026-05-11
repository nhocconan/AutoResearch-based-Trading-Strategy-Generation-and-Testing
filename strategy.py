#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeConfirm"
timeframe = "12h"
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
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels from previous 1d bar's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous 1d bar's range
    range_1d = high_1d - low_1d
    
    # Calculate Camarilla R3 and S3 levels
    camarilla_r3 = close_1d + (range_1d * 1.1 / 2)
    camarilla_s3 = close_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe (using previous 1d bar's values)
    r3_12h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_1d = close_1d_series.ewm(span=34, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: current volume > 2.0x 20-period average (tightened for 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND above 1d EMA34 (uptrend) AND volume surge
            if close[i] > r3_12h[i] and close[i] > ema_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND below 1d EMA34 (downtrend) AND volume surge
            elif close[i] < s3_12h[i] and close[i] < ema_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below S3 OR below 1d EMA34 (trend change)
            if close[i] < s3_12h[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above R3 OR above 1d EMA34 (trend change)
            if close[i] > r3_12h[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals