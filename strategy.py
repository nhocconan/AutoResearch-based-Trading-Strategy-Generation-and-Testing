#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot R3/S3 Breakout with 1d EMA34 Trend Filter and Volume Confirmation
# Long when: price breaks above R3 AND close > EMA34(1d) AND volume > 1.5x 20-period MA
# Short when: price breaks below S3 AND close < EMA34(1d) AND volume > 1.5x 20-period MA
# Exit when: price reverts to Pivot Point (PP) OR volume drops below average
# Uses Camarilla pivots for institutional levels, EMA34 for trend filter, volume for conviction
# Timeframe: 12h, HTF: 1d for EMA34. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Camarilla_R3S3_1dEMA34_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation on 12h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for EMA34 and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # need sufficient data for EMA34
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Calculate Camarilla pivot levels for 1d
    # Based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Range = High - Low
    range_1d = high_1d - low_1d
    # Resistance levels
    r1 = pp + (range_1d * 1.1 / 12)
    r2 = pp + (range_1d * 1.1 / 6)
    r3 = pp + (range_1d * 1.1 / 4)
    r4 = pp + (range_1d * 1.1 / 2)
    # Support levels
    s1 = pp - (range_1d * 1.1 / 12)
    s2 = pp - (range_1d * 1.1 / 6)
    s3 = pp - (range_1d * 1.1 / 4)
    s4 = pp - (range_1d * 1.1 / 2)
    
    # Align 1d indicators to 12h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND above EMA34 AND volume filter
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 AND below EMA34 AND volume filter
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reverts to Pivot Point OR volume drops below average
            if (close[i] <= pp_aligned[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reverts to Pivot Point OR volume drops below average
            if (close[i] >= pp_aligned[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals