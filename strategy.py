#!/usr/bin/env python3
"""
Hypothesis: 12-hour Camarilla pivot breakout with volume confirmation and weekly trend filter.
Trades only during high-volume breakouts at key pivot levels (R3/S3) in the direction of the weekly EMA(34).
Uses weekly EMA as trend filter to avoid counter-trend trades in bear markets.
Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.
Works in both bull and bear markets by aligning with weekly trend.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12-hour data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (R3, S3) from previous 12h bar
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pivot levels based on previous 12h bar
    R3 = close_12h + 1.1 * (high_12h - low_12h)
    S3 = close_12h - 1.1 * (high_12h - low_12h)
    
    # Align pivot levels to 12h timeframe (they're already at 12h frequency)
    R3_aligned = align_htf_to_ltf(prices, df_12h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_12h, S3)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 12-hour volume for volume confirmation
    vol_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need pivot levels, volume MA, and weekly EMA
    start_idx = max(20, 34)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        r3 = R3_aligned[i]
        s3 = S3_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_12h_aligned[i]
        trend_1w = ema_34_1w_aligned[i]
        
        # Volume filter: volume > 1.8x 12h average (restrictive to reduce trades)
        vol_filter = vol_now > 1.8 * vol_ma
        
        # Entry conditions: Camarilla pivot breakout with volume and weekly trend alignment
        if position == 0:
            # Long: break above R3 + volume + weekly uptrend
            if close[i] > r3 and vol_filter and close[i] > trend_1w:
                signals[i] = size
                position = 1
            # Short: break below S3 + volume + weekly downtrend
            elif close[i] < s3 and vol_filter and close[i] < trend_1w:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below weekly EMA or S3 level
            if close[i] < trend_1w or close[i] < s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above weekly EMA or R3 level
            if close[i] > trend_1w or close[i] > r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1wTrendFilter_Volume"
timeframe = "12h"
leverage = 1.0