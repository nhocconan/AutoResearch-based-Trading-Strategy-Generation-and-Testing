#!/usr/bin/env python3
"""
4h_1d_camarilla_volume_trend_v1
Uses 1d Camarilla pivot levels for entry/exit with volume confirmation and trend filter.
Only enters when price touches S3 or R3 levels (strong support/resistance) with volume spike.
Trend filter uses 12h EMA(20) to avoid counter-trend trades.
Designed for low trade frequency (target: 20-30 trades/year) to minimize fee drag.
Works in both bull and bear markets by fading extremes at pivot levels with trend alignment.
"""

name = "4h_1d_camarilla_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    range_ = high - low
    if range_ == 0:
        return close, close, close, close, close, close, close, close
    c = close
    s1 = c - (range_ * 1.0 / 12)
    s2 = c - (range_ * 2.0 / 12)
    s3 = c - (range_ * 3.0 / 12)
    s4 = c - (range_ * 4.0 / 12)
    r1 = c + (range_ * 1.0 / 12)
    r2 = c + (range_ * 2.0 / 12)
    r3 = c + (range_ * 3.0 / 12)
    r4 = c + (range_ * 4.0 / 12)
    return s1, s2, s3, s4, r1, r2, r3, r4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pre-calculate Camarilla levels for each day
    s1_1d = np.full_like(close_1d, np.nan)
    s2_1d = np.full_like(close_1d, np.nan)
    s3_1d = np.full_like(close_1d, np.nan)
    s4_1d = np.full_like(close_1d, np.nan)
    r1_1d = np.full_like(close_1d, np.nan)
    r2_1d = np.full_like(close_1d, np.nan)
    r3_1d = np.full_like(close_1d, np.nan)
    r4_1d = np.full_like(close_1d, np.nan)
    
    for i in range(len(df_1d)):
        s1, s2, s3, s4, r1, r2, r3, r4 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        s1_1d[i] = s1
        s2_1d[i] = s2
        s3_1d[i] = s3
        s4_1d[i] = s4
        r1_1d[i] = r1
        r2_1d[i] = r2
        r3_1d[i] = r3
        r4_1d[i] = r4
    
    # Align Camarilla levels to 4h timeframe
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_4h = align_htf_to_ltf(prices, df_1d, s4_1d)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_4h = align_htf_to_ltf(prices, df_1d, r2_1d)
    r3_4h = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_4h = align_htf_to_ltf(prices, df_1d, r4_1d)
    
    # Trend filter: 12h EMA(20)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(s3_4h[i]) or np.isnan(r3_4h[i]) or 
            np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price touches S3 with volume and above 12h EMA (uptrend)
        if (close[i] <= s3_4h[i] * 1.001 and close[i] >= s3_4h[i] * 0.999 and  # touches S3
            vol_confirm[i] and 
            close[i] > ema_12h_aligned[i] and  # uptrend filter
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price touches R3 with volume and below 12h EMA (downtrend)
        elif (close[i] >= r3_4h[i] * 0.999 and close[i] <= r3_4h[i] * 1.001 and  # touches R3
              vol_confirm[i] and 
              close[i] < ema_12h_aligned[i] and  # downtrend filter
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions: price moves back to S2/R2 or touches opposite level
        elif position == 1 and (close[i] >= s2_4h[i] or close[i] <= s4_4h[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] <= r2_4h[i] or close[i] >= r4_4h[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals