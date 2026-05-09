#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Previous 12h bar's OHLC (for Camarilla calculation)
    prev_close_12h = df_12h['close'].shift(1).values
    prev_high_12h = df_12h['high'].shift(1).values
    prev_low_12h = df_12h['low'].shift(1).values
    
    # Calculate Camarilla levels
    camarilla_pivot_12h = (prev_high_12h + prev_low_12h + prev_close_12h) / 3
    camarilla_range_12h = prev_high_12h - prev_low_12h
    camarilla_r3_12h = camarilla_pivot_12h + camarilla_range_12h * 1.1 / 4
    camarilla_s3_12h = camarilla_pivot_12h - camarilla_range_12h * 1.1 / 4
    
    # Align Camarilla levels to 6h
    camarilla_pivot_6h = align_htf_to_ltf(prices, df_12h, camarilla_pivot_12h)
    camarilla_r3_6h = align_htf_to_ltf(prices, df_12h, camarilla_r3_12h)
    camarilla_s3_6h = align_htf_to_ltf(prices, df_12h, camarilla_s3_12h)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: above 2.0x 12-period average (12*6h = 3 days)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 12  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_6h[i]) or np.isnan(camarilla_s3_6h[i]) or 
            np.isnan(ema_50_6h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma[i]  # Volume confirmation
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long breakout: price breaks above camarilla R3 with 12h uptrend
            if (close[i] > camarilla_r3_6h[i] and 
                close[i] > ema_50_6h[i] and  # 12h uptrend
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below camarilla S3 with 12h downtrend
            elif (close[i] < camarilla_s3_6h[i] and 
                  close[i] < ema_50_6h[i] and  # 12h downtrend
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below camarilla pivot (mean reversion)
            if close[i] < camarilla_pivot_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above camarilla pivot (mean reversion)
            if close[i] > camarilla_pivot_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals