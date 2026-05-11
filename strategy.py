#!/usr/bin/env python3
"""
6h_1d_Camarilla_R3_S3_Breakout_With_1dTrend_Volume
Hypothesis: On 6-hour timeframe, buy breakouts above Camarilla R3 and sell breakdowns below S3,
but only when aligned with 1-day trend (close > EMA34) and volume > 1.5x average.
This combines intraday breakout logic with daily trend filter and volume confirmation.
In bull markets: buys breakouts in uptrend. In bear markets: sells breakdowns in downtrend.
Volume filter ensures participation. Target: 12-30 trades/year per symbol.
"""

name = "6h_1d_Camarilla_R3_S3_Breakout_With_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d EMA34 for trend filter ---
    close_1d = df_1d['close'].values
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        alpha = 2 / (34 + 1)
        ema_34_1d[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_34_1d[i-1]
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- 1d volume average (20-period) for volume filter ---
    vol_1d = df_1d['volume'].values
    vol_avg_20_1d = np.full(len(vol_1d), np.nan)
    if len(vol_1d) >= 20:
        for i in range(19, len(vol_1d)):
            vol_avg_20_1d[i] = np.mean(vol_1d[i-19:i+1])
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # --- Calculate Camarilla levels from previous 1-day OHLC ---
    # Camarilla: based on previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values  # same as close_1d
    
    # Calculate levels for each day, then shift by 1 to get previous day's levels
    range_1d = high_1d - low_1d
    # Camarilla levels
    R3 = close_1d_prev + range_1d * 1.1 / 6
    S3 = close_1d_prev - range_1d * 1.1 / 6
    R4 = close_1d_prev + range_1d * 1.1 / 2
    S4 = close_1d_prev - range_1d * 1.1 / 2
    
    # Shift to get previous day's levels (today's levels based on yesterday's data)
    R3_prev = np.roll(R3, 1)
    S3_prev = np.roll(S3, 1)
    R4_prev = np.roll(R4, 1)
    S4_prev = np.roll(S4, 1)
    # First day has no previous
    R3_prev[0] = np.nan
    S3_prev[0] = np.nan
    R4_prev[0] = np.nan
    S4_prev[0] = np.nan
    
    # Align Camarilla levels to 6h
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3_prev)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3_prev)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4_prev)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4_prev)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 and volume average
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current volume > 1.5x 20-day average
        volume_filter = volume[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        if position == 0:
            # Long: break above R3 with uptrend and volume
            if (close[i] > R3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with downtrend and volume
            elif (close[i] < S3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter):
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: close below S3 or trend reverses
                if close[i] < S3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: close above R3 or trend reverses
                if close[i] > R3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals