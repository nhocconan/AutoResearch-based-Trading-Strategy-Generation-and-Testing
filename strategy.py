#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla R3/S3 levels from daily timeframe act as strong support/resistance.
# Breakout above R3 or below S3 on 12h triggers entry only when aligned with 1d trend (EMA34).
# Volume confirmation filters out false breakouts. Works in bull via breakouts above resistance
# and in bear via breakdowns below support. Low trade frequency due to multi-timeframe
# confirmation and precise entry levels.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close, close, close
    c = close
    h = high
    l = low
    r3 = c + (range_val * 1.1 / 2)
    r2 = c + (range_val * 1.1 / 4)
    r1 = c + (range_val * 1.1 / 6)
    s1 = c - (range_val * 1.1 / 6)
    s2 = c - (range_val * 1.1 / 4)
    s3 = c - (range_val * 1.1 / 2)
    return r3, r2, r1, s1, s2, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels for previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r3_1d, r2_1d, r1_1d, s1_1d, s2_1d, s3_1d = calculate_camarilla(
        high_1d, low_1d, close_1d
    )
    
    # Shift levels by 1 to use previous day's levels (no look-ahead)
    r3_1d = np.roll(r3_1d, 1)
    r2_1d = np.roll(r2_1d, 1)
    r1_1d = np.roll(r1_1d, 1)
    s1_1d = np.roll(s1_1d, 1)
    s2_1d = np.roll(s2_1d, 1)
    s3_1d = np.roll(s3_1d, 1)
    # Set first day's levels to NaN (no previous day)
    r3_1d[0] = np.nan
    r2_1d[0] = np.nan
    r1_1d[0] = np.nan
    s1_1d[0] = np.nan
    s2_1d[0] = np.nan
    s3_1d[0] = np.nan
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 12h data for entry signals
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Camarilla (1) + EMA (34) + vol EMA (20)
    start_idx = max(1, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter
        volume_filter = volume[i] > vol_ema20[i] * 1.5
        
        if position == 0:
            # Long: price breaks above R3 AND above 1d EMA34 AND volume confirmation
            if close[i] > r3_1d_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND below 1d EMA34 AND volume confirmation
            elif close[i] < s3_1d_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below S3 OR below 1d EMA34
            if close[i] < s3_1d_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above R3 OR above 1d EMA34
            if close[i] > r3_1d_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals