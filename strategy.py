#!/usr/bin/env python3
"""
Hypothesis: 12-hour Camarilla R3/S3 breakout with 1-week trend filter and volume confirmation.
Trades only during high-volume breakouts in the direction of the 1-week trend.
Designed to work in both bull and bear markets by using the 1-week trend as filter.
Target: 12-37 trades/year per symbol (48-148 total over 4 years) to minimize fee drift.
"""
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
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1-week EMA(50) for trend
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1-day data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (R3, S3) from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r3 = pivot + (range_1d * 1.1 / 2)
    s3 = pivot - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels (use previous day's levels for current 12h bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3, additional_delay_bars=1)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3, additional_delay_bars=1)
    
    # Get 12-hour data for volume filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12-hour volume MA(20)
    vol_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Camarilla levels, volume MA, and 1w EMA
    start_idx = max(2, 20, 50)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma_20_12h_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_12h_aligned[i]
        trend_1w = ema_50_1w_aligned[i]
        
        # Volume filter: volume > 2.0x 12h average (strict to reduce trades)
        vol_filter = vol_now > 2.0 * vol_ma
        
        # Entry conditions: Camarilla R3/S3 breakout with volume and 1w trend alignment
        if position == 0:
            # Long: break above R3 + volume + 1w uptrend
            if close[i] > r3_level and vol_filter and close[i] > trend_1w:
                signals[i] = size
                position = 1
            # Short: break below S3 + volume + 1w downtrend
            elif close[i] < s3_level and vol_filter and close[i] < trend_1w:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below 1w EMA or S3 level
            if close[i] < trend_1w or close[i] < s3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above 1w EMA or R3 level
            if close[i] > trend_1w or close[i] > r3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1wTrendFilter_Volume"
timeframe = "12h"
leverage = 1.0