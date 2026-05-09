#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 12h data for additional trend confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Previous 1d bar's OHLC (for Camarilla calculation)
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels R1 and S1 (inner bounds)
    camarilla_pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    camarilla_range_1d = prev_high_1d - prev_low_1d
    camarilla_r1_1d = camarilla_pivot_1d + camarilla_range_1d * 1.1 / 12
    camarilla_s1_1d = camarilla_pivot_1d - camarilla_range_1d * 1.1 / 12
    
    # Align Camarilla levels to 4h
    camarilla_pivot_4h = align_htf_to_ltf(prices, df_1d, camarilla_pivot_1d)
    camarilla_r1_4h = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_4h = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h EMA50 for additional trend confirmation
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_4h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: above 1.8x 20-period average (20*4h = 80h ≈ 3.3 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_4h[i]) or np.isnan(camarilla_s1_4h[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(ema_50_12h_4h[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.8 * vol_ma[i]  # Volume confirmation
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long breakout: price breaks above camarilla R1 with daily uptrend
            if (close[i] > camarilla_r1_4h[i] and 
                close[i] > ema_50_4h[i] and 
                close[i] > ema_50_12h_4h[i] and  # dual timeframe trend confirmation
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below camarilla S1 with daily downtrend
            elif (close[i] < camarilla_s1_4h[i] and 
                  close[i] < ema_50_4h[i] and 
                  close[i] < ema_50_12h_4h[i] and  # dual timeframe trend confirmation
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below camarilla pivot (mean reversion)
            if close[i] < camarilla_pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above camarilla pivot (mean reversion)
            if close[i] > camarilla_pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals