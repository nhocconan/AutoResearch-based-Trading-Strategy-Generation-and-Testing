#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_Pivot_Breakout_v1
Hypothesis: Use Camarilla pivot levels on 4h and 1d for institutional support/resistance,
with volume confirmation on 1h for breakout confirmation. Long when price breaks above
4h R3 or 1d R3 with volume > 1.5x 20-period average, short when breaks below S3 levels.
Camarilla levels provide tight, statistically significant levels that work in both
trending and ranging markets. Volume filter reduces false breakouts. Designed for
low trade frequency (target: 60-150 total over 4 years) to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_Camarilla_Pivot_Breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous period's OHLC for Camarilla calculation (4h)
    prev_high_4h = df_4h['high'].iloc[-2] if len(df_4h) >= 2 else df_4h['high'].iloc[-1]
    prev_low_4h = df_4h['low'].iloc[-2] if len(df_4h) >= 2 else df_4h['low'].iloc[-1]
    prev_close_4h = df_4h['close'].iloc[-2] if len(df_4h) >= 2 else df_4h['close'].iloc[-1]
    
    # Previous period's OHLC for Camarilla calculation (1d)
    prev_high_1d = df_1d['high'].iloc[-2] if len(df_1d) >= 2 else df_1d['high'].iloc[-1]
    prev_low_1d = df_1d['low'].iloc[-2] if len(df_1d) >= 2 else df_1d['low'].iloc[-1]
    prev_close_1d = df_1d['close'].iloc[-2] if len(df_1d) >= 2 else df_1d['close'].iloc[-1]
    
    # Calculate Camarilla levels (4h)
    range_4h = prev_high_4h - prev_low_4h
    if range_4h <= 0:
        camarilla_r3_4h = camarilla_s3_4h = 0
    else:
        camarilla_r3_4h = prev_close_4h + range_4h * 1.1 / 4  # R3 = Close + (Range * 1.1/4)
        camarilla_s3_4h = prev_close_4h - range_4h * 1.1 / 4  # S3 = Close - (Range * 1.1/4)
    
    # Calculate Camarilla levels (1d)
    range_1d = prev_high_1d - prev_low_1d
    if range_1d <= 0:
        camarilla_r3_1d = camarilla_s3_1d = 0
    else:
        camarilla_r3_1d = prev_close_1d + range_1d * 1.1 / 4  # R3 = Close + (Range * 1.1/4)
        camarilla_s3_1d = prev_close_1d - range_1d * 1.1 / 4  # S3 = Close - (Range * 1.1/4)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_4h_array = np.full(len(df_4h), camarilla_r3_4h)
    camarilla_s3_4h_array = np.full(len(df_4h), camarilla_s3_4h)
    camarilla_r3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h_array)
    camarilla_s3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h_array)
    
    camarilla_r3_1d_array = np.full(len(df_1d), camarilla_r3_1d)
    camarilla_s3_1d_array = np.full(len(df_1d), camarilla_s3_1d)
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d_array)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d_array)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = volume_series / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values  # default to 1.0 if no MA
    
    # Session filter: 08-20 UTC (active trading hours)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if outside session or any data invalid
        if not in_session[i] or \
           (np.isnan(camarilla_r3_4h_aligned[i]) or np.isnan(camarilla_s3_4h_aligned[i]) or
            np.isnan(camarilla_r3_1d_aligned[i]) or np.isnan(camarilla_s3_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Breakout conditions with volume filter and session
        long_breakout = (close[i] > camarilla_r3_4h_aligned[i] or close[i] > camarilla_r3_1d_aligned[i]) and vol_ratio[i] > 1.5
        short_breakout = (close[i] < camarilla_s3_4h_aligned[i] or close[i] < camarilla_s3_1d_aligned[i]) and vol_ratio[i] > 1.5
        
        # Exit conditions: return to opposite Camarilla level (S3 for long, R3 for short)
        long_exit = close[i] < camarilla_s3_4h_aligned[i] or close[i] < camarilla_s3_1d_aligned[i]
        short_exit = close[i] > camarilla_r3_4h_aligned[i] or close[i] > camarilla_r3_1d_aligned[i]
        
        # Signal logic
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals