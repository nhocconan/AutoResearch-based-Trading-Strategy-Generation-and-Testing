#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots (primary signal)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].values
    low_prev = df_1d['low'].values
    close_prev = df_1d['close'].values
    
    # Camarilla R3, S3 levels (most significant)
    camarilla_high = np.zeros(len(df_1d))
    camarilla_low = np.zeros(len(df_1d))
    camarilla_high[0] = np.nan  # first day has no previous
    camarilla_low[0] = np.nan
    
    for i in range(1, len(df_1d)):
        # Previous day's range
        range_prev = high_prev[i-1] - low_prev[i-1]
        camarilla_high[i] = close_prev[i-1] + 1.1 * range_prev / 2  # R3
        camarilla_low[i] = close_prev[i-1] - 1.1 * range_prev / 2   # S3
    
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4h EMA(34) for trend
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Get 1d data for volume filter
    vol_1d = df_1d['volume'].values
    vol_ma_10_1d = pd.Series(vol_1d).rolling(window=10, min_periods=10).mean().values
    vol_ma_10_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_10_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Camarilla, EMA, volume MA
    start_idx = max(1, 34, 10)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma_10_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        camarilla_high_val = camarilla_high_aligned[i]
        camarilla_low_val = camarilla_low_aligned[i]
        ema_trend = ema_34_4h_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_10_1d_aligned[i]
        
        # Volume filter: volume > 1.2x 1d MA (volume confirmation)
        vol_filter = vol_now > 1.2 * vol_ma
        
        # Entry conditions: Camarilla touch with volume and trend alignment
        if position == 0:
            # Long: touch S3 + volume + price above EMA (uptrend)
            if close[i] <= camarilla_low_val and vol_filter and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: touch R3 + volume + price below EMA (downtrend)
            elif close[i] >= camarilla_high_val and vol_filter and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches midpoint or trend reverses
            midpoint = (camarilla_high_val + camarilla_low_val) / 2
            if close[i] >= midpoint or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price reaches midpoint or trend reverses
            midpoint = (camarilla_high_val + camarilla_low_val) / 2
            if close[i] <= midpoint or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_CamarillaS3R3_Touch_EMA34_VolumeFilter"
timeframe = "4h"
leverage = 1.0