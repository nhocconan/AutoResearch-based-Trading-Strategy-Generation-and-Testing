#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R3S3_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Previous 1d bar's OHLC (for Camarilla calculation)
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    camarilla_pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    camarilla_range_1d = prev_high_1d - prev_low_1d
    camarilla_r3_1d = camarilla_pivot_1d + camarilla_range_1d * 1.1 / 4
    camarilla_s3_1d = camarilla_pivot_1d - camarilla_range_1d * 1.1 / 4
    
    # Align Camarilla levels to 1h
    camarilla_pivot_1h = align_htf_to_ltf(prices, df_1d, camarilla_pivot_1d)
    camarilla_r3_1h = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1h = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # 4h EMA20 for trend filter
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1h = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Volume filter: above 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    # Pre-compute session filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_1h[i]) or np.isnan(camarilla_s3_1h[i]) or 
            np.isnan(ema_20_1h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        in_session = (8 <= hours[i] <= 20)
        
        if position == 0:
            # Long breakout: price breaks above camarilla R3 with 4h uptrend
            if (close[i] > camarilla_r3_1h[i] and 
                close[i] > ema_20_1h[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.20
                position = 1
            # Short breakdown: price breaks below camarilla S3 with 4h downtrend
            elif (close[i] < camarilla_s3_1h[i] and 
                  close[i] < ema_20_1h[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below camarilla pivot
            if close[i] < camarilla_pivot_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price rises back above camarilla pivot
            if close[i] > camarilla_pivot_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals