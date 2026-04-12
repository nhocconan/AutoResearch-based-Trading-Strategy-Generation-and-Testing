#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Breakout_v1
Hypothesis: Use daily Camarilla pivot levels with breakout momentum on 4h timeframe.
Long when price breaks above R3 with volume confirmation, short when breaks below S3.
Incorporates weekly trend filter (ADX) to align with higher timeframe momentum.
Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag.
Works in bull via breakouts above resistance, in bear via breakdowns below support.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Pivot_Breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].iloc[-2] if len(df_1d) >= 2 else df_1d['high'].iloc[-1]
    prev_low = df_1d['low'].iloc[-2] if len(df_1d) >= 2 else df_1d['low'].iloc[-1]
    prev_close = df_1d['close'].iloc[-2] if len(df_1d) >= 2 else df_1d['close'].iloc[-1]
    
    # Calculate Camarilla pivot levels
    range_val = prev_high - prev_low
    if range_val <= 0:
        return np.zeros(n)
    
    # Camarilla levels: R3, S3
    camarilla_r3 = prev_close + range_val * 1.1 / 2
    camarilla_s3 = prev_close - range_val * 1.1 / 2
    
    # Align daily levels to 4h timeframe
    camarilla_r3_array = np.full(len(df_1d), camarilla_r3)
    camarilla_s3_array = np.full(len(df_1d), camarilla_s3)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_array)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_array)
    
    # Weekly ADX for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX (14-period)
    plus_dm = np.zeros(len(high_1w))
    minus_dm = np.zeros(len(high_1w))
    tr = np.zeros(len(high_1w))
    
    for i in range(1, len(high_1w)):
        plus_dm[i] = max(high_1w[i] - high_1w[i-1], 0)
        minus_dm[i] = max(low_1w[i-1] - low_1w[i], 0)
        tr[i] = max(high_1w[i] - low_1w[i], 
                   abs(high_1w[i] - close_1w[i-1]), 
                   abs(low_1w[i] - close_1w[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr14 = wilder_smooth(tr, period)
    plus_dm14 = wilder_smooth(plus_dm, period)
    minus_dm14 = wilder_smooth(minus_dm, period)
    
    # Avoid division by zero
    divisor = np.where(tr14 != 0, tr14, 1e-10)
    plus_di14 = 100 * plus_dm14 / divisor
    minus_di14 = 100 * minus_dm14 / divisor
    
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 
                  0)
    adx = wilder_smooth(dx, period)
    
    # Align weekly ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = volume_series / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any data invalid
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with volume and trend filter
        long_setup = (high[i] > camarilla_r3_aligned[i] and 
                     vol_ratio[i] > 1.5 and 
                     adx_aligned[i] > 25)
        short_setup = (low[i] < camarilla_s3_aligned[i] and 
                      vol_ratio[i] > 1.5 and 
                      adx_aligned[i] > 25)
        
        # Exit conditions: return to opposite Camarilla level or trend weakening
        long_exit = (low[i] < camarilla_s3_aligned[i] or 
                    adx_aligned[i] < 20)
        short_exit = (high[i] > camarilla_r3_aligned[i] or 
                     adx_aligned[i] < 20)
        
        # Signal logic
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals