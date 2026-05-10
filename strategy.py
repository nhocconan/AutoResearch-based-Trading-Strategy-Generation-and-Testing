#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSurge
# Hypothesis: Camarilla pivot levels (R1, S1) act as strong support/resistance in trending markets.
# In strong daily trends (price > EMA50_1d for longs, price < EMA50_1d for shorts),
# price retracing to Camarilla R1/S1 levels offers high-probability continuation entries.
# Volume surge confirms institutional interest in the breakout.
# Works in bull markets (follows uptrend continuations) and bear markets (follows downtrend continuations).
# Uses only 3 core conditions: trend filter, pivot retracement, volume surge.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSurge"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous day's OHLC
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C, H, L are from previous daily bar
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan  # First day has no previous day
    
    # Calculate Camarilla levels
    rang = prev_high - prev_low
    r1 = prev_close + rang * 1.1 / 12
    s1 = prev_close - rang * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: 20-period MA on 4h chart (~3.3 days)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50_1d (50), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_surge = volume[i] > volume_ma[i] * 1.5
        
        # Price retracement to Camarilla levels with breakout
        # For long: price crosses above R1 from below in uptrend
        # For short: price crosses below S1 from above in downtrend
        if i > 0:
            cross_above_r1 = (close[i] > r1_aligned[i]) and (close[i-1] <= r1_aligned[i-1])
            cross_below_s1 = (close[i] < s1_aligned[i]) and (close[i-1] >= s1_aligned[i-1])
        else:
            cross_above_r1 = False
            cross_below_s1 = False
        
        if position == 0:
            # Long entry: uptrend + retracement to R1 + volume surge
            if uptrend and cross_above_r1 and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + retracement to S1 + volume surge
            elif downtrend and cross_below_s1 and volume_surge:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price reaches S1 (opposite level)
            if not uptrend or (close[i] < s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price reaches R1 (opposite level)
            if not downtrend or (close[i] > r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals